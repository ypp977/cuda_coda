#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf() 绝对值函数
#include <fstream> // 文件输入输出流（写CSV）
#include <iostream>
#include <vector>

#define BLOCK_SIZE 32 // 每个block的线程块大小：32x32
#define TOL 1e-5f     // 误差容忍度（float比较时使用）

// CUDA错误检查函数
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// cuBLAS错误检查函数
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ----------------------------------------------
// 手写的最基础版SGEMM核函数 (没有共享内存优化)
// 计算 C = α * A × B + β * C
// ----------------------------------------------
__global__ void mysgemm_v0(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // 当前线程负责计算矩阵C中的哪个元素
    int gx = blockIdx.x * blockDim.x + threadIdx.x; // C的列索引
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // C的行索引

    // 边界检查（避免越界访问）
    if (gx >= N || gy >= N)
    {
        return;
    }

    // tmp 用于存储当前元素的点积结果
    float tmp = 0.0f;

    // 按行乘法求和：C[gy][gx] = sum(A[gy][i] * B[i][gx])
    for (int i = 0; i < K; i++)
    {
        tmp += A[gy * K + i] * B[i * N + gx];
    }

    // 写回结果：α * tmp + β * C
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

// ----------------------------------------------
// 主函数：运行性能测试并输出结果
// ----------------------------------------------
int main()
{
    // 要测试的矩阵规模列表（方阵）
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    // 打开CSV文件，写入表头
    std::ofstream csv_file("sgemm_benchmark_v1.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // 遍历每个矩阵尺寸
    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        // 为 NxN 矩阵分配内存
        size_t size = N * N * sizeof(float);
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_v1 = (float*)malloc(size);

        // 分配GPU内存
        float *device_A, *device_B, *device_C_V1;
        checkCudaError(cudaMalloc(&device_A, size), "cudaMalloc device_A failed");
        checkCudaError(cudaMalloc(&device_B, size), "cudaMalloc device_B failed");
        checkCudaError(cudaMalloc(&device_C_V1, size), "cudaMalloc device_C_V1 failed");

        bool out_of_memory = false;

        try
        {
            // 初始化A和B矩阵
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // 把主机数据拷贝到GPU
            checkCudaError(cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_A failed");
            checkCudaError(cudaMemcpy(device_B, host_B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_B failed");

            // 创建cuBLAS句柄
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            // 设置矩阵乘法参数 α 和 β
            float alpha = 1.0f;
            float beta = 0.0f;

            // 创建CUDA事件用于计时
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // -----------------------------
            // (1) cuBLAS SGEMM 预热（warm-up）
            // -----------------------------
            int warpup_time = 10;
            for (int i = 0; i < warpup_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // 不转置
                                             N, N, N,                          // 矩阵维度
                                             &alpha, device_B, N,              // B矩阵与其步长
                                             device_A, N,                      // A矩阵与其步长
                                             &beta, device_C_V1, N),           // 输出矩阵C与步长
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize(); // 等待预热结束

            // -----------------------------
            // (2) cuBLAS 实测部分
            // -----------------------------
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_B, N, device_A, N, &beta, device_C_V1, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 将 cuBLAS 结果拷回主机
            checkCudaError(cudaMemcpy(host_C_cublas, device_C_V1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_C_cublas failed");

            // -----------------------------
            // (3) 手写 SGEMM 核函数测试
            // -----------------------------
            checkCudaError(cudaMemset(device_C_V1, 0, size), "cudaMemset device_C_V1 failed");

            // 配置线程块
            dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
            dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

            // 预热
            for (int i = 0; i < warpup_time; i++)
            {
                mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, device_A, device_B, beta,
                                                device_C_V1);
            }
            cudaDeviceSynchronize();

            // 正式计时
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start mysgemm_v1) failed");
            for (int i = 0; i < repeat_time; i++)
            {
                mysgemm_v0<<<blocks, threads>>>(N, N, N, alpha, device_A, device_B, beta,
                                                device_C_V1);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop mysgemm_v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize mysgemm_v1 failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime mysgemm_v1 failed");

            // 将结果拷回主机
            checkCudaError(cudaMemcpy(host_C_v1, device_C_V1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_C_v1 failed");

            // -----------------------------
            // (4) 验证结果正确性
            // -----------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_v1[i]) > TOL)
                {
                    error_count++;
                }
            }

            // -----------------------------
            // (5) 计算性能指标（GFlops）
            // -----------------------------
            // 理论计算量为 2 * N^3（乘加运算各一次）
            // cublas_time/v1_time 单位是毫秒，因此乘 1e6 转为秒
            float cublas_gflops = repeat_time * N * N * N * 2.0f / (cublas_time * 1e6f);
            float v1_gflops = repeat_time * N * N * N * 2.0f / (v1_time * 1e6f);

            // 写入CSV结果
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // -----------------------------
            // (6) 清理资源
            // -----------------------------
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaFree(device_A);
            cudaFree(device_B);
            cudaFree(device_C_V1);

            free(host_A);
            free(host_B);
            free(host_C_cublas);
            free(host_C_v1);
        }
        catch (...)
        {
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        // 输出状态
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
