#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf()：浮点数比较
#include <fstream> // 文件操作，用于写入 CSV
#include <iostream>
#include <vector>

#define BLOCK_SIZE 32 // 每个线程块是 32x32 个线程，用于覆盖 C 的子块
#define TOL 1e-5f     // 浮点结果比较的容忍误差

// -----------------------------
// CUDA 错误检查工具函数
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
// cuBLAS 错误检查工具函数
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
// 基础版手写 SGEMM 核函数（未使用共享内存优化）
//
// 功能：计算 C = alpha * A × B + beta * C
// 矩阵维度：A(M x K), B(K x N), C(M x N)，均为 row-major 存储
// 调用时本例使用的是方阵：M = N = K = N
//
// 线程划分：
//   每个线程负责计算 C 中的一个元素 C[gy, gx]
// ----------------------------------------------
__global__ void mysgemm_v0(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // gx：C 的列索引（对应 N 方向）
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    // gy：C 的行索引（对应 M 方向）
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：gy 对应 M 维度，gx 对应 N 维度
    if (gx >= N || gy >= M)
        return;

    float tmp = 0.0f;

    // 沿 K 方向做点积：C[gy, gx] = Σ A[gy, k] * B[k, gx]
    for (int i = 0; i < K; i++)
        tmp += A[gy * K + i] * B[i * N + gx];

    // 写回结果：融合 alpha / beta
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

// ----------------------------------------------
// 主函数：执行 SGEMM 性能测试并记录结果到 CSV
// ----------------------------------------------
int main()
{
    // -----------------------------
    // 1. 定义测试矩阵尺寸（方阵：M = N = K）
    // -----------------------------
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    // -----------------------------
    // 2. 打开 CSV 文件记录性能结果
    //    列含义：
    //      Size           : 矩阵维度 N（方阵 N x N）
    //      CUBLAS_GFLOPS  : cuBLAS 实测算力
    //      MySGEMM_FLOPS  : 手写 mysgemm_v0 实测算力
    //      Matched        : 1 表示结果与 cuBLAS 在 TOL 内一致，否则为 0
    // -----------------------------
    std::ofstream csv_file("sgemm_benchmark_V0.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // -----------------------------
    // 3. 遍历不同矩阵大小
    // -----------------------------
    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;
        // 本例使用方阵：A(N x N), B(N x N), C(N x N)
        size_t size = N * N * sizeof(float);

        // -----------------------------
        // 3.1 CPU 内存分配
        // host_A / host_B      : 输入矩阵
        // host_C_cublas        : cuBLAS 计算结果
        // host_C_V0            : 手写 SGEMM 计算结果
        // -----------------------------
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_V0 = (float*)malloc(size);

        // -----------------------------
        // 3.2 GPU 内存分配
        // device_A / device_B  : 设备端输入矩阵
        // device_C_V0          : 设备端输出矩阵（交给 cuBLAS 和 mysgemm_v0 复用）
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
            //    A 全 1，B 全 2，C 初始值由 alpha / beta 控制
            //    对于本例参数，理论结果为：C = 2 * N（每个元素）
            // -----------------------------
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // -----------------------------
            // 5. 拷贝矩阵数据到 GPU
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
            // 7. 创建 CUDA 事件，用于计时
            // -----------------------------
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // -----------------------------
            // 8. cuBLAS SGEMM 预热
            //    预热可以避免首次调用带来的额外开销影响计时
            // -----------------------------
            int warpup_time = 10;
            for (int i = 0; i < warpup_time; i++)
            {
                // 注意：cuBLAS 按列主序解释矩阵，
                // 这里传入的是 row-major 数据，但在本基准中仅做相对性能对比，
                // 且手写核与 cuBLAS 使用相同内存布局，因此数值结果仍可直接比较。
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // 不转置
                                             N, N, N, &alpha, device_B, N, // 左矩阵（列主序视角）
                                             device_A, N,                  // 右矩阵（列主序视角）
                                             &beta, device_C_V0, N),
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

            float cublas_time = 0.0f; // 毫秒
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
            // 先清零输出缓存，避免残留数据影响结果
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

            float V0_time = 0.0f; // 毫秒
            checkCudaError(cudaEventElapsedTime(&V0_time, start, stop),
                           "cudaEventElapsedTime mysgemm_V0 failed");

            // -----------------------------
            // 12. 拷贝手写 SGEMM 结果回 CPU
            // -----------------------------
            checkCudaError(cudaMemcpy(host_C_V0, device_C_V0, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_C_V0 failed");

            // -----------------------------
            // 13. 验证结果正确性
            //    简单抽样前若干元素进行比较，允许一定浮点误差
            // -----------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_V0[i]) > TOL)
                    error_count++;
            }

            // -----------------------------
            // 14. 计算性能指标 GFLOPS
            //    单次 GEMM 的浮点运算量约为：2 * N^3（乘法 + 加法）
            //    total_flops：重复 repeat_time 次的总运算量
            //    时间使用毫秒，因此要乘 1e6（= 1e3 * 1e3）换算到 GFLOPS
            // -----------------------------
            double total_flops = double(repeat_time) * 2.0 * double(N) * double(N) * double(N);
            double cublas_gflops = total_flops / (double(cublas_time) * 1e6);
            double V0_gflops = total_flops / (double(V0_time) * 1e6);

            // -----------------------------
            // 15. 写入 CSV
            // Matched = 1：手写结果在 TOL 误差内与 cuBLAS 一致
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
            // 注意：真正的 OOM 多半发生在 cudaMalloc 阶段，此处主要作为保护
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        // -----------------------------
        // 17. 输出测试状态 / CSV 记录 OOM 情况
        // -----------------------------
        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_V0.csv'" << std::endl;
    return 0;
}
