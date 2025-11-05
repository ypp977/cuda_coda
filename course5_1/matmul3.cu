#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------
mysgemm_v3: 高性能手写矩阵乘法 kernel（C = alpha * A * B + beta * C）
------------------------------------------

模板参数：
    BLOCK_M, BLOCK_N, BLOCK_K : 每个 thread block 处理的子矩阵大小
    THREAD_M, THREAD_N        : 每个线程计算的输出子块大小

函数参数：
    M, N, K       : 矩阵维度 (A[M×K], B[K×N], C[M×N])
    alpha, beta   : 缩放系数（与 BLAS 规范一致）
    A, B          : 输入矩阵（行主序 Row-major）
    C             : 输出矩阵

算法设计思路：
    1. 将矩阵按 tile（子块）分块，每个 block 计算 C 的 BLOCK_M × BLOCK_N 子矩阵。
    2. 每个线程在 tile 内负责一个 THREAD_M × THREAD_N 的局部计算。
    3. 使用共享内存缓存 A、B 的 tile，减少全局内存访问。
    4. 使用寄存器缓存线程加载的 A/B 数据，并在寄存器中累加计算结果。
    5. 沿 K 方向逐块累积局部计算结果到寄存器 accum。
    6. 最后将寄存器结果按 alpha、beta 系数写回全局内存 C。
------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, const float* __restrict__ A,
                           const float* __restrict__ B, float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // block_row_idx: block 在 C 中的行编号
    // block_col_idx: block 在 C 中的列编号
    // ------------------------------
    const int block_col_idx = blockIdx.x;
    const int block_row_idx = blockIdx.y;

    // ------------------------------
    // 2. Tile 内线程划分
    // thread_per_row/col: tile 内线程数量
    // thread_per_block: tile 内总线程数量
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    // ------------------------------
    // 3. 当前线程在 tile 内负责计算的局部 C 子块起点（左上角坐标）
    // local_row_idx / local_col_idx 表示线程在 tile 内的局部偏移
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M;

    // ------------------------------
    // 4. 分配共享内存用于缓存当前 tile 的 A/B 子块
    // shared_a: BLOCK_K × BLOCK_M，用于缓存 A tile 并可转置访问
    // shared_b: BLOCK_K × BLOCK_N，用于缓存 B tile
    // ------------------------------
    __shared__ float shared_a[BLOCK_K * BLOCK_M];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算每个线程需要加载的 float4 向量数
    // 向量化加载可以减少全局内存访问次数
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 计算线程加载 A/B 的行列起点和步长
    // a_load_row/col, b_load_row/col: 当前线程在 tile 内加载起点
    // a_load_stride/b_load_stride: 跨线程加载步长，保证 tile 全覆盖
    // ------------------------------
    const int a_load_row = threadIdx.x / (BLOCK_K / 4);
    const int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int a_load_stride = BLOCK_M / vec4_load_per_thread_a;

    const int b_load_row = threadIdx.x / (BLOCK_N / 4);
    const int b_load_col = (threadIdx.x % (BLOCK_N / 4)) * 4;
    const int b_load_stride = BLOCK_K / vec4_load_per_thread_b;

    // ------------------------------
    // 7. 寄存器缓存
    // accum: 存放每个线程计算的 C 子块结果
    // reg_a_vec: 缓存线程加载的 A 向量
    // reg_a_tile / reg_b_tile: 当前迭代从共享内存取出的 A/B 元素
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.0f};
    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.};
    float reg_a_tile[THREAD_M];
    float reg_b_tile[THREAD_N];

    // ------------------------------
    // 8. 将 A/B/C 指针偏移到当前 block 对应的子矩阵起始位置
    // ------------------------------
    A = &A[block_row_idx * BLOCK_M * K];                           // 当前 block A 子块起始行
    B = &B[block_col_idx * BLOCK_N];                               // 当前 block B 子块起始列
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N]; // 输出子块起始位置

    // ------------------------------
    // 9. 沿 K 方向循环处理 BLOCK_K 大小的 tile
    // ------------------------------
#pragma unroll
    for (int k_block_start = 0; k_block_start < K; k_block_start += BLOCK_K)
    {
        // ---- 9.1 将 A tile 从全局内存加载到共享内存并转置 ----
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            const int reg_idx = i / a_load_stride * 4;
            FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

            // 将 A tile 转置存入共享内存，便于连续访存
            shared_a[OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
            shared_a[OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
            shared_a[OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
            shared_a[OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
        }

        // ---- 9.2 将 B tile 从全局内存加载到共享内存 ----
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            FETCH_FLOAT4(shared_b[OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
        }

        __syncthreads(); // 保证所有线程完成共享内存加载

        // ---- 9.3 移动全局指针到下一 K 块 ----
        A += BLOCK_K;     // A 往右移动 BLOCK_K 列
        B += BLOCK_K * N; // B 往下移动 BLOCK_K 行

        // ---- 9.4 在共享内存中执行矩阵乘法 ----
#pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            // 加载 A tile 当前行到寄存器
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[m]) =
                    FETCH_FLOAT4(shared_a[OFFSET(k_inner, local_row_idx + m, BLOCK_M)]);
            }

            // 加载 B tile 当前列到寄存器
#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[n]) =
                    FETCH_FLOAT4(shared_b[OFFSET(k_inner, local_col_idx + n, BLOCK_N)]);
            }

            // 在寄存器中累加 C 子块
#pragma unroll
            for (int m = 0; m < THREAD_M; m++)
            {
#pragma unroll
                for (int n = 0; n < THREAD_N; n++)
                {
                    accum[m][n] += reg_a_tile[m] * reg_b_tile[n];
                }
            }
        }
        __syncthreads(); // 保证累加完成后再进行下一 K tile
    }

    // ------------------------------
    // 10. 将累加结果写回全局内存
    // ------------------------------
#pragma unroll
    for (int m = 0; m < THREAD_M; m++)
    {
#pragma unroll
        for (int n = 0; n < THREAD_N; n += 4)
        {
            float4 c_val = FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]);
            // 按 BLAS 规范计算 alpha * A*B + beta * C
            c_val.x = alpha * accum[m][n] + beta * c_val.x;
            c_val.y = alpha * accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * accum[m][n + 3] + beta * c_val.w;
            FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]) = c_val;
        }
    }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes()
{
    return {4096};
}
int main()
{
    int device_id = 7;
    checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::vector<int> sizes = generateSizes();
    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v1 = (float*)malloc(size);

        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;

        try
        {
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i)
            {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            // 拷贝到设备
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy B to device failed");

            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");
            float alpha = 1.0f;
            float beta = 0.0f;

            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // warmup
            int warpup_time = 10; // 热身次数
            for (int i = 0; i < warpup_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cuBLAS SGEMM
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }

            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 拷贝 cuBLAS 结果
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_cublas failed");

            // mysgemm_v1
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");
            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");
            // 结果比较
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i)
            {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
                {
                    error_count++;
                }
            }

            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);         // GFlops
            // 写入CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v1);

            free(A);
            free(B);
            free(C_cublas);
            free(C_v1);
        }
        catch (...)
        {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (!out_of_memory)
        {
            std::cout << "Finished size: " << N << std::endl;
        }
        else
        {
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'" << std::endl;
    return 0;
}
