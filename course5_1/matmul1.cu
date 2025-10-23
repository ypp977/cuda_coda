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

// 模板参数：BLOCK_SIZE 表示线程块的边长（即 tile 的大小）
// 例如 BLOCK_SIZE = 32，则每个线程块处理 32×32 的子矩阵
template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // 当前线程块在网格中的二维索引
    int bx = blockIdx.x; // 对应输出矩阵 C 的列方向块索引
    int by = blockIdx.y; // 对应输出矩阵 C 的行方向块索引

    // tile 尺寸定义（方便阅读）
    const int BM = BLOCK_SIZE; // tile 高度（对应 A、C 的行数）
    const int BN = BLOCK_SIZE; // tile 宽度（对应 B、C 的列数）
    const int BK = BLOCK_SIZE; // tile 深度（对应 A 的列数、B 的行数）

    // 每个线程在 block 内的一维索引转为二维坐标
    // 假设一个 block 含有 BLOCK_SIZE * BLOCK_SIZE 个线程
    int tx = threadIdx.x % BN; // 当前线程在 tile 内的列索引
    int ty = threadIdx.x / BN; // 当前线程在 tile 内的行索引

    // 为 A 和 B 分配共享内存，用于存放当前 tile 的子块
    // 共享内存能显著减少全局内存访问，提高带宽利用率
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 将全局指针 A、B、C 偏移到当前 tile 的起始位置
    // 注意：
    //   A 的维度为 M×K
    //   B 的维度为 K×N
    //   C 的维度为 M×N
    A = &A[by * BM * K];           // 当前 block 对应的 A 子矩阵起始位置
    B = &B[bx * BN];               // 当前 block 对应的 B 子矩阵起始位置
    C = &C[by * BM * N + bx * BN]; // 当前 block 对应的 C 子矩阵起始位置

    // 每个线程用于计算 C 子矩阵中一个元素的临时累积结果
    float tmp = 0.0f;

    // 分块循环（tile-by-tile 遍历 K 维）
    // 每次加载 A、B 各一个子块到共享内存
    for (int k = 0; k < K; k += BK)
    {
        // 每个线程负责将 A 子矩阵和 B 子矩阵的一部分加载到共享内存中
        // A 的维度：BM×BK，B 的维度：BK×BN
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];

        // 同步所有线程，确保共享内存中的数据已加载完毕
        __syncthreads();

        // 更新 A、B 指针到下一个 tile
        A += BK;     // A 向右平移 BK 列
        B += BK * N; // B 向下平移 BK 行

        // 当前 tile 内进行计算：矩阵乘法的核心逻辑
        // tmp 累加每一行·列的乘积结果
        for (int i = 0; i < BK; i++)
        {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }

        // 再次同步，确保共享内存不会被下一个 tile 的数据覆盖
        __syncthreads();
    }

    // 计算完成后，将结果写回全局内存
    // C = alpha * (A × B) + beta * C
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
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
                mysgemm_v2<32><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }

            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");

            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_v2<32><<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
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
