#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf() 用于浮点数比较
#include <fstream> // 文件操作，用于写CSV
#include <iostream>
#include <vector>

#define BLOCK_SIZE 32 // 每个线程块的尺寸：32x32
#define TOL 1e-5f     // 浮点误差容忍度

// -----------------------------
// CUDA 错误检查函数
// -----------------------------
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------
// cuBLAS 错误检查函数
// -----------------------------
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------
// 手写最基础版 SGEMM 核函数（未优化共享内存）
// 计算公式：C = α * A × B + β * C
// 每个线程计算 C 矩阵中的一个元素
// ----------------------------------------------
__global__ void mysgemm_v0(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // 行索引

    if (gx >= N || gy >= N) // 边界检查
        return;

    float tmp = 0.0f;

    // 计算点积
    for (int i = 0; i < K; i++)
        tmp += A[gy * K + i] * B[i * N + gx];

    // 写回 C
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

// ----------------------------------------------
// 主函数：执行性能测试并写入CSV
// ----------------------------------------------
int main()
{
    // -----------------------------
    // 1. 定义待测试矩阵尺寸（方阵）
    // -----------------------------
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    // -----------------------------
    // 2. 打开 CSV 文件，用于保存性能结果
    // -----------------------------
    std::ofstream csv_file("sgemm_benchmark_V0.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // -----------------------------
    // 3. 遍历每个矩阵大小进行测试
    // -----------------------------
    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);

        // -----------------------------
        // 3.1 CPU 内存分配
        // host_A/B: 输入矩阵
        // host_C_cublas: cuBLAS 输出
        // host_C_V0: 手写 SGEMM 输出
        // -----------------------------
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_V0 = (float*)malloc(size);

        // -----------------------------
        // 3.2 GPU 内存分配
        // -----------------------------
        float *device_A, *device_B, *device_C_V0;
        checkCudaError(cudaMalloc(&device_A, size), "cudaMalloc device_A failed");
        checkCudaError(cudaMalloc(&device_B, size), "cudaMalloc device_B failed");
        checkCudaError(cudaMalloc(&device_C_V0, size), "cudaMalloc device_C_V0 failed");

        bool out_of_memory = false;

        try
        {
            // -----------------------------
            // 4. 初始化矩阵数据
            // -----------------------------
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // -----------------------------
            // 5. 拷贝矩阵数据到GPU
            // -----------------------------
            checkCudaError(cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_A failed");
            checkCudaError(cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_B failed");

            // -----------------------------
            // 6. 创建 cuBLAS 句柄
            // -----------------------------
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            // -----------------------------
            // 7. 创建 CUDA 事件用于计时
            // -----------------------------
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // -----------------------------
            // 8. cuBLAS SGEMM 预热（warm-up）
            // -----------------------------
            int warpup_time = 10;
            for (int i = 0; i < warpup_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_B, N, device_A, N, &beta, device_C_V0, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // -----------------------------
            // 9. cuBLAS SGEMM 正式计时
            // -----------------------------
            int repeat_time = 50;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_B, N, device_A, N, &beta, device_C_V0, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // -----------------------------
            // 10. 拷贝 cuBLAS 结果回 CPU
            // -----------------------------
            checkCudaError(cudaMemcpy(host_C_cublas, device_C_V0, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_C_cublas failed");

            // -----------------------------
            // 11. 手写 SGEMM 预热与计时
            // -----------------------------
            checkCudaError(cudaMemset(device_C_V0, 0, size), "cudaMemset device_C_V0 failed");

            dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

            // 预热
            for (int i = 0; i < warpup_time; i++)
                mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, device_A, device_B, beta,
                                                device_C_V0);
            cudaDeviceSynchronize();

            // 正式计时
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start mysgemm_V0) failed");
            for (int i = 0; i < repeat_time; i++)
                mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, device_A, device_B, beta,
                                                device_C_V0);
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop mysgemm_V0) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize mysgemm_V0 failed");

            float V0_time = 0;
            checkCudaError(cudaEventElapsedTime(&V0_time, start, stop),
                           "cudaEventElapsedTime mysgemm_V0 failed");

            // -----------------------------
            // 12. 拷贝手写 SGEMM 结果回 CPU
            // -----------------------------
            checkCudaError(cudaMemcpy(host_C_V0, device_C_V0, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_C_V0 failed");

            // -----------------------------
            // 13. 验证结果正确性
            // -----------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_V0[i]) > TOL)
                    error_count++;
            }

            // -----------------------------
            // 14. 计算性能指标 GFLOPS
            // -----------------------------
            double total_flops = double(repeat_time) * 2.0 * double(N) * double(N) * double(N);
            double cublas_gflops = total_flops / (double(cublas_time) * 1e6);
            double V0_gflops = total_flops / (double(V0_time) * 1e6);

            // -----------------------------
            // 15. 写入 CSV
            // Matched = 1 表示手写结果与 cuBLAS 一致
            // -----------------------------
            csv_file << N << "," << cublas_gflops << "," << V0_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // -----------------------------
            // 16. 清理资源
            // -----------------------------
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_A);
            cudaFree(device_B);
            cudaFree(device_C_V0);
            free(host_A);
            free(host_B);
            free(host_C_cublas);
            free(host_C_V0);
        }
        catch (...)
        {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        // 输出当前测试状态
        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmarkv0.csv'" << std::endl;
    return 0;
}
