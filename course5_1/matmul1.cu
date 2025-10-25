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
mysgemm_v1: 手写矩阵乘法 Kernel
计算公式: C = alpha * A * B + beta * C
------------------------------------------
参数说明：
  M: 矩阵 A 的行数
  N: 矩阵 B 的列数（同时是矩阵 C 的列数）
  K: 矩阵 A 的列数 = 矩阵 B 的行数
  alpha: 矩阵乘法系数
  beta:  累加系数（控制是否叠加原有 C）
  A, B, C: 分别是输入矩阵和输出矩阵（行主序 Row-major）
  BLOCK_SIZE: 每个线程块计算的 tile 尺寸（方块区域大小）
*/
template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // ======== Block 级别坐标 ========
    // 当前 block 在输出矩阵 C 中的 tile 索引
    int block_row = blockIdx.y; // 沿着行方向的 tile 编号
    int block_col = blockIdx.x; // 沿着列方向的 tile 编号

    // ======== Tile 尺寸定义 ========
    // 每个 block 负责计算一个 BLOCK_M × BLOCK_N 的输出子块
    const int BLOCK_M = BLOCK_SIZE; // tile 的行数
    const int BLOCK_N = BLOCK_SIZE; // tile 的列数
    const int BLOCK_K = BLOCK_SIZE; // tile 内 K 方向分块宽度

    // ======== Thread 级别坐标 ========
    // 每个线程在 block 内二维坐标（行、列）
    int thread_row = threadIdx.y; // 线程在 tile 内的行索引
    int thread_col = threadIdx.x; // 线程在 tile 内的列索引

    // ======== 共享内存分配 ========
    // 用于存储从全局内存中加载的 A、B 子块
    __shared__ float shared_A[BLOCK_M * BLOCK_K];
    __shared__ float shared_B[BLOCK_K * BLOCK_N];

    // ======== 全局内存起始地址 ========
    // 当前 block 对应的 A、B、C 子块起始地址
    const float* A_start = &A[block_row * BLOCK_M * K]; // A 从第 block_row*BLOCK_M 行开始
    const float* B_start = &B[block_col * BLOCK_N];     // B 从第 block_col*BLOCK_N 列开始
    float* C_start = &C[block_row * BLOCK_M * N + block_col * BLOCK_N]; // C 对应 tile 的起始位置

    float result = 0.0f;

    for (int k_block = 0; k_block < K; k_block += BLOCK_K)
    {
        shared_A[threadIdx.y * BLOCK_K + threadIdx.x] = A_start[threadIdx.y * K + threadIdx.x];
    }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
int main()
{
    std::vector<int> sizes = {1024};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v2.csv");
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

            dim3 blockDim(1024);
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32));

            for (int i = 0; i < warpup_time; ++i)
            {
                mysgemm_v1<32><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_v1<32><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
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
