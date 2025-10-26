#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f // 浮点数误差容忍度

// ==============================
// CUDA 错误检查函数
// ==============================
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ==============================
// CUBLAS 错误检查函数
// ==============================
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
    // ==============================
    // Block 级别坐标
    // ==============================
    int block_row = blockIdx.y; // 当前 block 在 C 中的行方向 tile 索引
    int block_col = blockIdx.x; // 当前 block 在 C 中的列方向 tile 索引

    // ==============================
    // Tile 尺寸定义
    // ==============================
    const int BLOCK_M = BLOCK_SIZE; // tile 行数
    const int BLOCK_N = BLOCK_SIZE; // tile 列数
    const int BLOCK_K = BLOCK_SIZE; // tile K 方向厚度（每次沿 K 加载宽度）

    // ==============================
    // Thread 级别坐标
    // ==============================
    int thread_row = threadIdx.y; // 当前线程在 tile 内的行索引
    int thread_col = threadIdx.x; // 当前线程在 tile 内的列索引

    // ==============================
    // 共享内存分配
    // ==============================
    __shared__ float shared_A[BLOCK_M * BLOCK_K]; // 存储 A 子块
    __shared__ float shared_B[BLOCK_K * BLOCK_N]; // 存储 B 子块

    // ==============================
    // 全局内存起始地址
    // ==============================
    const float* A_start = &A[block_row * BLOCK_M * K];                 // A 当前 tile 行的起始行
    const float* B_start = &B[block_col * BLOCK_N];                     // B 当前 tile 列的起始列
    float* C_start = &C[block_row * BLOCK_M * N + block_col * BLOCK_N]; // C 当前 tile 起点

    // ==============================
    // 每个线程累积结果
    // ==============================
    float result = 0.0f; // 对应 C_tile 的一个元素

    // ==============================
    // 沿 K 方向分块循环
    // ==============================
    for (int k_block = 0; k_block < K; k_block += BLOCK_K)
    {
        // ------------------------------
        // 1. 将 A、B 当前 tile 加载到共享内存
        // ------------------------------
        shared_A[thread_row * BLOCK_K + thread_col] = A_start[thread_row * K + thread_col];
        shared_B[thread_row * BLOCK_N + thread_col] = B_start[thread_row * N + thread_col];

        // 等待所有线程完成加载
        __syncthreads();

        // ------------------------------
        // 2. 在共享内存中完成部分矩阵乘法
        // ------------------------------
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            result +=
                shared_A[thread_row * BLOCK_K + k_inner] * shared_B[k_inner * BLOCK_N + thread_col];
        }

        // 等待所有线程完成计算，准备加载下一块
        __syncthreads();

        // ------------------------------
        // 3. 移动到下一 K 块
        // ------------------------------
        A_start += BLOCK_K;     // A 按列向右移动 BLOCK_K 列
        B_start += BLOCK_K * N; // B 按行向下移动 BLOCK_K 行（每行 N 元素）
    }

    // ==============================
    // 写回结果到全局内存
    // ==============================
    C_start[thread_row * N + thread_col] =
        alpha * result + beta * C_start[thread_row * N + thread_col];
}

// ==============================
// 宏定义：向上取整
// ==============================
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main()
{
    // ==============================
    // 测试矩阵大小
    // ==============================
    std::vector<int> sizes = {1024}; // 这里只测试 1024x1024

    std::ofstream csv_file("sgemm_benchmark_v1.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);

        // ==============================
        // Host 内存分配
        // ==============================
        float* host_a = (float*)malloc(size);
        float* host_b = (float*)malloc(size);
        float* host_c_cublas = (float*)malloc(size);
        float* host_c_v1 = (float*)malloc(size);

        // ==============================
        // Device 内存分配
        // ==============================
        float *device_a, *device_b, *device_c_v1;
        checkCudaError(cudaMalloc(&device_a, size), "cuda malloc device_a error");
        checkCudaError(cudaMalloc(&device_b, size), "cuda malloc device_b error");
        checkCudaError(cudaMalloc(&device_c_v1, size), "cuda malloc device_c_v1 error");

        bool out_of_memory = false;

        try
        {
            // ==============================
            // 初始化 Host 矩阵
            // ==============================
            for (int i = 0; i < N * N; i++)
            {
                host_a[i] = 1.0f; // A 全 1
                host_b[i] = 2.0f; // B 全 2
            }

            // 复制 Host -> Device
            checkCudaError(cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_a to device error");
            checkCudaError(cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_b to device error");

            // ==============================
            // CUBLAS handle 创建
            // ==============================
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate error");

            float alpha = 1.0f, beta = 0.0f;

            // ==============================
            // CUDA 计时事件
            // ==============================
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start error");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop error");

            int warmup_time = 10; // 预热次数
            for (int i = 0; i < warmup_time; i++)
            {
                // CUBLAS SGEMM 计算
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_b, N, device_a, N, &beta, device_c_v1, N),
                                 "cublasSgemm error");
            }
            cudaDeviceSynchronize();

            // ==============================
            // CUBLAS Benchmark
            // ==============================
            int repeat_time = 50;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start error");
            for (int i = 0; i < repeat_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_b, N, device_a, N, &beta, device_c_v1, N),
                                 "cublasSgemm error");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop error");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop error");

            float cublas_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime error");

            checkCudaError(cudaMemcpy(host_c_cublas, device_c_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v1 to host_c_cublas error");

            // ==============================
            // MySGEMM 预热
            // ==============================
            checkCudaError(cudaMemset(device_c_v1, 0, size), "cudaMemset device_c_v1 error");

            dim3 blockDim(32, 32);                          // 每个 block 32x32 threads
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32)); // grid size

            for (int i = 0; i < warmup_time; i++)
            {
                mysgemm_v1<32>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v1);
            }
            cudaDeviceSynchronize();

            // ==============================
            // MySGEMM Benchmark
            // ==============================
            checkCudaError(cudaMemset(device_c_v1, 0, size), "cudaMemset device_c_v1 error");

            checkCudaError(cudaEventRecord(start), "cudaEventRecord start error");
            for (int i = 0; i < repeat_time; i++)
            {
                mysgemm_v1<32>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v1);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop error");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop error");

            float v1_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime error");

            checkCudaError(cudaMemcpy(host_c_v1, device_c_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v1 to host_c_v1 error");

            // ==============================
            // 精度验证
            // ==============================
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_c_cublas[i] - host_c_v1[i]) > TOL)
                {
                    error_count++;
                }
            }

            // ==============================
            // 计算 GFLOPS
            // ==============================
            // 每次矩阵乘法有 2*N^3 次浮点操作（N^3 乘 + N^3 加）
            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);

            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // ==============================
            // 释放资源
            // ==============================
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_a);
            cudaFree(device_b);
            cudaFree(device_c_v1);
            free(host_a);
            free(host_b);
            free(host_c_cublas);
            free(host_c_v1);
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
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v1.csv'" << std::endl;
    return 0;
}
