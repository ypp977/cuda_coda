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
    N: 矩阵 B 的列数（同时也是矩阵 C 的列数）
    K: 矩阵 A 的列数 = 矩阵 B 的行数
    alpha: 矩阵乘法系数
    beta:  累加系数（控制是否叠加原有 C）
    A, B, C: 输入输出矩阵（行主序 Row-major）

矩阵维度：
    A: M × K
    B: K × N
    C: M × N

算法思路：
    1. 将矩阵按 tile（子块）分块。
    2. 每个 block 负责计算一个 BLOCK_SIZE × BLOCK_SIZE 的 C 子矩阵。
    3. 每次从全局内存加载 A、B 对应的 tile 到共享内存 shared_A / shared_B。
    4. 在共享内存中完成部分乘加运算。
    5. 沿着 K 方向逐块累积，最终得到 C_tile。
*/

template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // ======== Block 级别坐标 ========
    int block_row = blockIdx.y; // 当前 block 对应 C 的行方向 tile 索引
    int block_col = blockIdx.x; // 当前 block 对应 C 的列方向 tile 索引

    // ======== Tile 尺寸定义 ========
    // 一个 block 负责计算一个 BLOCK_SIZE × BLOCK_SIZE 的 C 子块
    const int BLOCK_M = BLOCK_SIZE;
    const int BLOCK_N = BLOCK_SIZE;
    const int BLOCK_K = BLOCK_SIZE; // 每次沿 K 方向加载的宽度（tile 的厚度）

    // ======== Thread 级别坐标 ========
    int thread_row = threadIdx.y; // 当前线程在 tile 内的行索引
    int thread_col = threadIdx.x; // 当前线程在 tile 内的列索引

    // ======== 共享内存分配 ========
    // 存储从全局内存加载的 A、B 子块，大小为 BLOCK_M×BLOCK_K 和 BLOCK_K×BLOCK_N
    __shared__ float shared_A[BLOCK_M * BLOCK_K];
    __shared__ float shared_B[BLOCK_K * BLOCK_N];

    // ======== 全局内存起始地址 ========
    // 计算当前 block 对应的 A、B、C 子块在全局内存中的起点

    // 对 A：
    // block_row 表示第几个 A 子块行，每个子块行高为 BLOCK_M
    // 因此偏移量为 block_row * BLOCK_M * K
    const float* A_start = &A[block_row * BLOCK_M * K];

    // 对 B：
    // block_col 表示第几个 B 子块列，每列宽为 BLOCK_N
    // 因此偏移量为 block_col * BLOCK_N
    const float* B_start = &B[block_col * BLOCK_N];

    // 对 C：
    // 每个子块在 C 中的偏移量由行偏移和列偏移共同决定
    float* C_start = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ======== 累积结果寄存器 ========
    // 每个线程对应 C_tile 的一个元素
    float result = 0.0f;

    // ======== 沿 K 方向分块循环 ========
    // 每次循环处理 A 的 BLOCK_K 列、B 的 BLOCK_K 行
    for (int k_block = 0; k_block < K; k_block += BLOCK_K)
    {
        // -------------------------------
        // 1. 从全局内存加载 A、B 当前 tile 到共享内存
        // -------------------------------
        // A_tile 对应 A 中第 block_row 个 tile 行，第 k_block 个 tile 列
        // 每个线程负责加载一个元素
        shared_A[thread_row * BLOCK_K + thread_col] = A_start[thread_row * K + thread_col];

        // B_tile 对应 B 中第 k_block 个 tile 行，第 block_col 个 tile 列
        shared_B[thread_row * BLOCK_N + thread_col] = B_start[thread_row * N + thread_col];

        // 等待所有线程完成加载
        __syncthreads();

        // -------------------------------
        // 2. 在共享内存中完成矩阵乘法部分计算
        // -------------------------------
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            result +=
                shared_A[thread_row * BLOCK_K + k_inner] * shared_B[k_inner * BLOCK_N + thread_col];
        }

        // 等待所有线程完成计算，准备加载下一块
        __syncthreads();

        // -------------------------------
        // 3. 移动到下一 tile 段
        // -------------------------------
        // 对 A：向右移动 BLOCK_K 列
        A_start += BLOCK_K;

        // 对 B：向下移动 BLOCK_K 行，每行有 N 个元素
        // 因为 B 是 K×N 按行主序存储，所以跳过 BLOCK_K*N 个元素
        B_start += BLOCK_K * N;
    }

    // ======== 结果写回全局内存 ========
    // 每个线程写一个元素
    C_start[thread_row * N + thread_col] =
        alpha * result + beta * C_start[thread_row * N + thread_col];
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
