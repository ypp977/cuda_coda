#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf 用于浮点数比较
#include <fstream> // 文件输出流，用于写CSV
#include <iostream>
#include <vector>

// CUDA kernel 参数
#define BLOCK_SIZE 128 // 每个 block 的线程数
#define TOL 1e-5f      // 计算结果误差容忍度

// 检查 CUDA API 返回值，如果出错则打印信息并退出
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 返回值，如果出错则打印信息并退出
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 矩阵偏移宏，用于访问行列数据
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将连续的四个 float 视为 float4，方便向量化加载/存储
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/*
------------------------------------------------------------
mysgemm_v4: 手写高性能SGEMM kernel (C = alpha * A * B + beta * C)
------------------------------------------------------------

模板参数:
    BM, BN, BK : Block级别tile大小
    TM, TN     : 每个线程负责的子块大小

函数参数:
    M, N, K : 矩阵维度 (A[MxK], B[KxN], C[MxN])
    alpha, beta : 缩放系数
    A, B      : 输入矩阵 (Row-major)
    C         : 输出矩阵

算法核心:
1. 将矩阵按tile分块，每个block处理C的BMxBN子矩阵。
2. 每个线程计算TMxTN的局部子矩阵。
3. 使用共享内存缓存A/B tile，减少全局内存访问。
4. 寄存器缓存线程加载的A/B数据，并累加到C子块。
5. 沿K方向逐块累积。
6. 最后将寄存器结果按alpha/beta写回全局内存。
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void __launch_bounds__(256)
    mysgemm_v4(int M, int N, int K, float alpha, float* __restrict__ A, float* __restrict__ B,
               float beta, float* __restrict__ C)
{
    const int block_col_idx = blockIdx.x;
    const int block_row_idx = blockIdx.y;

    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    const int local_col_idx = (threadIdx.x / thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x % thread_per_row) * THREAD_M;

    __shared__ float shared_a[2][BLOCK_K * BLOCK_M];
    __shared__ float shared_b[2][BLOCK_K * BLOCK_N];

    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    const int a_load_row = threadIdx.x / (BLOCK_K / 4);
    const int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int a_load_stride = BLOCK_M / vec4_load_per_thread_a;

    const int b_load_row = threadIdx.x / (BLOCK_K / 4);
    const int b_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int b_load_stride = BLOCK_N / vec4_load_per_thread_b;

    float accum[THREAD_M][THREAD_N] = {0.};

    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.};
    float reg_b_vec[4 * vec4_load_per_thread_b] = {0.};
    float reg_b_tile[2][THREAD_M];
    float reg_a_tile[2][THREAD_N];

    A = &A[block_row_idx * BLOCK_M * K];
    B = &B[block_col_idx * BLOCK_N];
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N];

#pragma unroll
    for (int i = 0; i < BLOCK_M; i += a_load_stride)
    {
        int reg_idx = i / a_load_stride * 4;
        FETCH_FLOAT4(reg_a_vec[reg_idx]) = FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

        shared_a[0][OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
        shared_a[0][OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
        shared_a[0][OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
        shared_a[0][OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_K; i += b_load_stride)
    {
        FETCH_FLOAT4(shared_b[0][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
            FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
    }
    __syncthreads();

#pragma unroll
    for (int m = 0; m < THREAD_M; m += 4)
    {
        FETCH_FLOAT4(reg_a_tile[0][m]) =
            FETCH_FLOAT4(shared_a[0][OFFSET(0, local_row_idx + m, BLOCK_M)]);
    }

#pragma unroll
    for (int n = 0; n < THREAD_N; n += 4)
    {
        FETCH_FLOAT4(reg_b_tile[0][n]) =
            FETCH_FLOAT4(shared_b[0][OFFSET(0, local_col_idx + n, BLOCK_N)]);
    }

    int write_index = 1;
    int load_index;
    int k = 0;

    do
    {
        K += BLOCK_K;
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = i / a_load_stride * 4;
                FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BLOCK_K; i += b_load_stride)
            {
                int reg_idx = i / b_load_stride * 4;
                FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                    FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
            }
        }

        load_index = write_index;

#pragma unroll
        for (int bk = 0; bk < BLOCK_K - 1; bk++)
        {
#pragma unroll
            for (int m = 0; m < THREAD_M; m++)
            {
#pragma unroll
                for (int n = 0; n < THREAD_N; n++)
                {
                    accum[m][n] += reg_a_tile[bk % 2][m] * reg_b_tile[bk % 2][n];
                }
            }
        }

        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = i / a_load_stride * 4;
                shared_a[write_index][OFFSET(a_load_col, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx];
                shared_a[write_index][OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 1];
                shared_a[write_index][OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 2];
                shared_a[write_index][OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 3];
            }
#pragma unroll
            for (int i = 0; i < BLOCK_K; i += b_load_stride)
            {
                int reg_idx = i / b_load_stride * 4;
                FETCH_FLOAT4(shared_b[write_index][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                    FETCH_FLOAT4(reg_b_vec[reg_idx]);
            }
            __syncthreads();
            write_index ^= 1;
        }
    } while (k < K)

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
// 整除向上宏
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// 生成测试矩阵大小
std::vector<int> generateSizes()
{
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256)
        sizes.push_back(i);
    return sizes;
}

int main()
{
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件用于记录性能测试结果
    std::ofstream csv_file("sgemm_benchmark_v7.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);

        // CPU 分配内存
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v1 = (float*)malloc(size);

        // GPU 分配内存
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

            // 拷贝数据到 GPU
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy B to device failed");

            // cuBLAS handle 创建
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // warmup 先执行几次，稳定 GPU 性能
            int warpup_time = 10;
            for (int i = 0; i < warpup_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cuBLAS SGEMM 性能测试
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

            // 拷贝 cuBLAS 结果回 CPU
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_cublas failed");

            // 手写 kernel 测试
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i)
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            cudaDeviceSynchronize();

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");
            for (int i = 0; i < repeat_time; ++i)
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");
            checkCudaError(cudaGetLastError(), "cuda get last error failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果回 CPU
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");

            // 检查结果是否匹配
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i)
            {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
                    error_count++;
            }

            // 计算 GFLOPS
            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);         // GFlops

            float ratio = v1_gflops / cublas_gflops;

            // 写入 CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

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
            cudaDeviceSynchronize();
        }
        catch (...)
        {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'" << std::endl;
    return 0;
}
