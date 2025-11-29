#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf()：浮点数绝对值，用于结果比较
#include <fstream> // 文件操作，用于写入 CSV
#include <iostream>
#include <vector>

/*
------------------------------------------------------------
模块名：基础版 SGEMM 基准测试（mysgemm_v0 vs cuBLAS）
------------------------------------------------------------
1. 功能概述：
   - 实现一个最朴素的 SGEMM 核函数 mysgemm_v0（无共享内存优化）：
       C = alpha * A × B + beta * C
   - 使用 cuBLAS Sgemm 作为参考实现：
       · 校验 mysgemm_v0 计算结果正确性
       · 对比两者的性能（GFLOPS）
   - 结果以 CSV（sgemm_benchmark_V0.csv）输出，便于后续画图和分析。

2. 存储布局约定：
   - A、B、C 在 host 与 device 上均按 row-major 存储：
       · A 的行跨度（leading dimension）为 K
       · B 的行跨度为 N
       · C 的行跨度为 N
   - cuBLAS 内部按 column-major 解释矩阵，本代码通过「交换 A/B 的位置」
     和「统一使用 N×N 方阵」使得数值结果在 row-major 空间下仍与手写核一致。

3. 测试矩阵与参数：
   - 矩阵规模：N ∈ {128, 256, 512, 1024, 2048, 4096, 8192}，统一使用方阵 N×N。
   - 初始化：
       · A 元素全为 1.0f
       · B 元素全为 2.0f
       · alpha = 1.0f, beta = 0.0f
     理论结果：C[i,j] = Σ_k A[i,k] * B[k,j] = Σ_k 1 * 2 = 2 * N。
   - 计时策略：
       · 每个实现先 warmup 多次，再重复 repeat_time 次计算并计时。

4. 性能计量：
   - 单次 GEMM 浮点运算量约为：2 * N^3（乘法 + 加法）。
   - 总 FLOPs = repeat_time * 2 * N^3。
   - 使用 cudaEventElapsedTime 得到毫秒 ms：
       GFLOPS = 总 FLOPs / (time_ms * 1e6)。

5. 结果字段：
   - Size          ：矩阵维度 N
   - CUBLAS_GFLOPS ：cuBLAS 实测算力
   - MySGEMM_FLOPS ：mysgemm_v0 实测算力
   - Matched       ：mysgemm_v0 与 cuBLAS 比较是否在 TOL 误差内（1/0）
------------------------------------------------------------
*/

#define BLOCK_SIZE 32 // 每个线程块使用 32×32 个线程，组成一个 BLOCK_SIZE×BLOCK_SIZE 的 C 子块
#define TOL 1e-5f     // 浮点结果比较的容忍误差（绝对误差阈值）

// ------------------------------------------------------------
// CUDA 错误检查工具函数
// ------------------------------------------------------------
// 用法：
//   - 所有 cudaXXX 调用后都通过该函数检查返回值。
//   - 一旦发现错误，立刻打印错误消息并退出程序，避免后续结果无意义。
// ------------------------------------------------------------
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ------------------------------------------------------------
// cuBLAS 错误检查工具函数
// ------------------------------------------------------------
// 用法：
//   - 所有 cublasXXX 调用后都通过该函数检查状态码。
//   - 出错时打印状态码（枚举值），方便快速定位哪个 cuBLAS 调用失败。
// ------------------------------------------------------------
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------------------------
模块名：mysgemm_v0（基础版 SGEMM 核函数）
------------------------------------------------------------
1. 功能：
   - 计算单精度矩阵乘法：
       C = alpha * A × B + beta * C
   - 矩阵维度：
       · A: M × K
       · B: K × N
       · C: M × N
     且均为 row-major 存储。

2. 线程映射策略：
   - 网格/线程块组织：
       · blockIdx.x / threadIdx.x → 对应 C 的列索引（N 方向）
       · blockIdx.y / threadIdx.y → 对应 C 的行索引（M 方向）
   - 每个线程负责计算 C 中的一个元素 C[gy, gx]，无共享内存，直接从全局内存读 A、B。

3. 访存与计算：
   - 对于每个输出元素 C[gy, gx]：
       · 在 K 维度上做一次标量点积：
           tmp = Σ_{i=0..K-1} A[gy, i] * B[i, gx]
       · 最终写回：
           C[gy, gx] = alpha * tmp + beta * C[gy, gx]

4. 特点：
   - 实现简单，易于验证正确性；
   - 访存模式未优化：
       · A 的访问在一个 warp 内相对规整（行方向连续）；
       · B 的访问在 K 维度上存在 stride，global memory 访问不够连续；
   - 作为 baseline 适合和后续共享内存优化版做对比。
------------------------------------------------------------
*/
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
    // row-major 线性下标：
    //   A[gy, i] → A[gy * K + i]
    //   B[i, gx] → B[i * N + gx]
    for (int i = 0; i < K; i++)
        tmp += A[gy * K + i] * B[i * N + gx];

    // 写回结果：融合 alpha / beta
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

/*
------------------------------------------------------------
模块名：main（SGEMM 基准测试驱动）
------------------------------------------------------------
1. 功能：
   - 对多个方阵尺寸 N×N：
       · 使用 cuBLAS Sgemm 进行基准计算与计时；
       · 使用手写 kernel mysgemm_v0 进行计算与计时；
       · 比较两者结果在容差 TOL 内是否一致；
       · 将 GFLOPS 与匹配情况写入 CSV 文件。

2. cuBLAS 与 row-major 的关系说明（重要）：
   - cuBLAS 假定输入矩阵为 column-major（列主序），即：
       线性下标：A_col(i, j) = A_data[i + j * lda]
   - 本代码中 device_A / device_B 实际按 row-major 布局：
       A_row(i, j) = A_data[i * N + j]
   - 在方阵 N×N 的场景下，若直接以 lda=N 方式传给 cuBLAS：
       · cuBLAS 看到的 A_col 实际是 A_row^T
       · cuBLAS 看到的 B_col 实际是 B_row^T
   - 本代码调用：
       C_cublas = B_col * A_col = B_row^T * A_row^T = (A_row * B_row)^T
   - 再将 C_cublas（column-major）当作 row-major 一维数组使用时，相当于对其转置一次，
     因此主机侧看到的 host_C_cublas 恰好等价于 A_row * B_row 的 row-major 存储。
   - 结论：
       - 虽然未显式指定转置标志位，利用「方阵 + 交换 A/B」的方式，
         仍然可以在 row-major 空间得到与手写核一致的数值结果。
------------------------------------------------------------
*/
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
    //      CUBLAS_GFLOPS  : cuBLAS 实测算力（GFLOPS）
    //      MySGEMM_FLOPS  : 手写 mysgemm_v0 实测算力（GFLOPS）
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
        size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

        // -----------------------------
        // 3.1 CPU 内存分配
        //   host_A / host_B   : 输入矩阵
        //   host_C_cublas     : cuBLAS 计算结果
        //   host_C_V0         : 手写 SGEMM 计算结果
        // -----------------------------
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_V0 = (float*)malloc(size);

        // -----------------------------
        // 3.2 GPU 内存分配
        //   device_A / device_B : 设备端输入矩阵
        //   device_C_V0         : 设备端输出矩阵（交给 cuBLAS 和 mysgemm_v0 复用）
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
            //    A 全 1，B 全 2
            //    理论结果（alpha=1,beta=0）为：C[i,j] = 2 * N
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
            //    目的：让 kernel / 驱动等进入稳定状态，避免首次调用偏慢
            // -----------------------------
            int warpup_time = 10; // 预热次数（命名保持原样，含义为 warmup 次数）
            for (int i = 0; i < warpup_time; i++)
            {
                // 调用说明（列主序视角）：
                //   C_cublas = B_col * A_col
                // 且 B_col、A_col 实际分别对应 row-major 的 B_row^T、A_row^T，
                // 结合 main 头部的说明，可得到 host 侧看到的结果为 A_row * B_row。
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // 不做转置
                                             N,                                // m
                                             N,                                // n
                                             N,                                // k
                                             &alpha, device_B, N, // 左矩阵（列主序视角）
                                             device_A, N,         // 右矩阵（列主序视角）
                                             &beta, device_C_V0, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // -----------------------------
            // 9. cuBLAS SGEMM 正式计时
            // -----------------------------
            int repeat_time = 50; // 正式计时重复次数
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
            // 先清零输出缓存，避免 cuBLAS 结果残留影响手写核的 beta*C 分支
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
            //    顺序扫描前 N*N 个元素，最多统计 10 个误差点
            //    判定条件：|C_cublas - C_V0| > TOL
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
            // 16. 清理资源（正常路径）
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
            // 注意：绝大多数 OOM 会在 cudaMalloc 阶段被检测到；
            // 此处 catch 只是防止极端情况下程序直接崩溃。
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
