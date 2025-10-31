#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f

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
mysgemm_v2: 分块矩阵乘法（Tiled Matrix Multiplication）
计算公式: C = alpha * A * B + beta * C
------------------------------------------

【参数说明】
    M: 矩阵 A 的行数
    N: 矩阵 B 的列数（同时也是矩阵 C 的列数）
    K: 矩阵 A 的列数 = 矩阵 B 的行数
    alpha: 矩阵乘法系数
    beta:  累加系数（控制是否叠加原有 C）
    A, B, C: 输入输出矩阵（行主序 Row-major）

【矩阵维度】
    A: M × K
    B: K × N
    C: M × N

【算法设计思路】
    1. 将矩阵按 tile（子块）分块，每个 block 计算 C 的一个 TILE_M × TILE_N 子矩阵。
    2. 每个线程在 tile 内负责一个 THREAD_M × THREAD_N 的局部计算区域。
    3. 使用共享内存缓存 A、B 的 tile，减少全局内存访问。
    4. 沿 K 方向逐块累积局部计算结果到寄存器 tmp。
    5. 最后将寄存器结果写回全局内存。
------------------------------------------
*/

template <const int TILE_M, const int TILE_N, const int TILE_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // block_row, block_col 确定了 C_tile 在大矩阵中的位置
    // ------------------------------
    int block_col = blockIdx.x; // block 负责的列 tile
    int block_row = blockIdx.y; // block 负责的行 tile

    // ------------------------------
    // 2. Tile 内线程划分
    // threads_per_tile_x/y 表示每个 tile 内线程块数
    // total_threads 是 tile 内总线程数
    // ------------------------------
    int threads_per_tile_x = TILE_N / THREAD_N; // 横向线程数量
    int threads_per_tile_y = TILE_M / THREAD_M; // 纵向线程数量
    int total_threads = threads_per_tile_x * threads_per_tile_y;

    // ------------------------------
    // 3. 当前线程在 tile 内负责的局部计算块起点
    // local_row/col 表示线程负责的 C_tile 局部子矩阵左上角
    // ------------------------------
    int local_col = (threadIdx.x % threads_per_tile_x) * THREAD_N;
    int local_row = (threadIdx.y % threads_per_tile_y) * THREAD_M;

    // ------------------------------
    // 4. 分配共享内存，用于缓存当前 tile 的 A、B 子块
    // shared_a: TILE_M × TILE_K
    // shared_b: TILE_K × TILE_N
    // ------------------------------
    __shared__ float shared_a[TILE_M * TILE_K];
    __shared__ float shared_b[TILE_K * TILE_N];

    // ------------------------------
    // 5. block 对应的全局内存起始地址
    // A_block_ptr/B_block_ptr 指向当前 tile 对应的全局 A/B
    // C_block_ptr 指向输出矩阵 C 的对应 tile
    // ------------------------------
    const float* A_block_ptr = A + block_row * TILE_M * K;
    const float* B_block_prt = B + block_col * TILE_N;
    float* C_block_ptr = C + block_row * TILE_M * N + block_col * TILE_N;

    // ------------------------------
    // 6. 定义每个线程加载 A、B 子块时的分工
    // threadIdx.x/y 映射到二维线程块布局
    // a_load_row/col、b_load_row/col 决定线程加载共享内存的位置
    // a_load_stride/b_load_stride 控制跨线程步长，保证 tile 全覆盖
    // ------------------------------
    int a_load_row = threadIdx.x / TILE_K;
    int a_load_col = threadIdx.x % TILE_K;
    int a_load_stride = total_threads / TILE_K;

    int b_load_row = threadIdx.y / TILE_N;
    int b_load_col = threadIdx.y % TILE_N;
    int b_load_stride = total_threads / TILE_N;

    // ------------------------------
    // 7. 局部寄存器缓存，用于保存线程负责的 C 子块结果
    // THREAD_M × THREAD_N 表示每个线程负责的计算块大小
    // ------------------------------
    float tmp[THREAD_M][THREAD_N] = {0.0f};

    // ------------------------------
    // 8. 主循环：沿 K 方向分块计算
    // 每次循环处理 TILE_K 深度的 A、B 子块
    // ------------------------------
#pragma unroll
    for (int k = 0; k < K; k += TILE_K)
    {
        // --------------------------
        // 8.1 从全局内存加载 A 子块到共享内存
        // 遍历 TILE_M 行，用 a_load_stride 跨线程步长避免重复加载
        // --------------------------
#pragma unroll
        for (int i = 0; i < TILE_M; i += a_load_stride)
        {
            shared_a[(a_load_row + i) * TILE_K + a_load_col] =
                A_block_ptr[(a_load_row + i) * K + a_load_col];
        }

        // --------------------------
        // 8.2 从全局内存加载 B 子块到共享内存
        // 类似 A 的加载策略
        // --------------------------
#pragma unroll
        for (int i = 0; i < TILE_M; i += b_load_stride)
        {
            shared_b[(b_load_row + i) * TILE_N + b_load_col] =
                B_block_prt[(b_load_row + i) * N + b_load_col];
        }

        // --------------------------
        // 8.3 同步线程，确保共享内存加载完毕
        // --------------------------
        __syncthreads();

        // --------------------------
        // 8.4 更新 A、B 全局指针到下一个 K 子块位置
        // --------------------------
        A_block_ptr += TILE_K;     // A 向右移动 TILE_K 列
        B_block_prt += TILE_K * N; // B 向下移动 TILE_K 行

        // --------------------------
        // 8.5 寄存器内乘加，累积 tmp
        // shared_a: TILE_M × TILE_K
        // shared_b: TILE_K × TILE_N
        // 每个线程计算 THREAD_M × THREAD_N 的结果
        // --------------------------
#pragma unroll
        for (int i = 0; i < TILE_K; i++)
        {
            for (int j = 0; j < THREAD_M; j++)
            {
                for (int l = 0; l < THREAD_N; l++)
                {
                    tmp[j][l] += shared_a[(local_row + j) * TILE_K + i] *
                                 shared_b[i * TILE_N + (local_col + l)];
                }
            }
        }
        __syncthreads(); // 等待所有线程完成本轮计算
    }

    // ------------------------------
    // 9. 将寄存器 tmp 中的结果写回全局内存 C
    // 注意考虑 alpha、beta 系数
    // ------------------------------
#pragma unroll
    for (int j = 0; j < THREAD_M; j++)
    {
        for (int l = 0; l < THREAD_N; l++)
        {
            int c_index = (local_row + j) * N + (local_col + l);
            C_block_ptr[c_index] = alpha * tmp[j][l] + beta * C_block_ptr[c_index];
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

            // mysgemm_v4
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
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
