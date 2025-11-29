#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf：用于浮点数误差比较
#include <fstream> // std::ofstream：用于写 CSV 文件
#include <iostream>
#include <vector>

#define TOL 1e-5f // 结果校验时允许的最大浮点误差

// ==============================
// CUDA 错误检查函数
// ------------------------------
// 用法：包装所有 cudaXXX 调用；
// 一旦返回值不是 cudaSuccess，就打印 msg + CUDA 错误字符串并直接退出。
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
// cuBLAS 错误检查函数
// ------------------------------
// 用法：包装所有 cublasXXX 调用；
// 一旦返回值不是 CUBLAS_STATUS_SUCCESS，就打印 msg + 状态码并直接退出。
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
------------------------------------------------------------
模块名：mysgemm_v1（使用共享内存 tiling 的基础 SGEMM kernel）
------------------------------------------------------------
1. 功能：
   - 使用 block-level tiling + shared memory 实现：
       C = alpha * A * B + beta * C
   - 矩阵维度（row-major）：
       A: M × K
       B: K × N
       C: M × N

2. 参数说明：
   - M          : 矩阵 A 的行数
   - N          : 矩阵 B / C 的列数
   - K          : 矩阵 A 的列数 = 矩阵 B 的行数
   - alpha      : 矩阵乘法缩放系数
   - beta       : 累加系数（控制是否叠加原有 C）
   - A, B, C    : 输入/输出矩阵指针（均按 row-major 存储）

3. 算法结构（典型 Block-level tiling）：
   1) C 被划分为 BLOCK_SIZE × BLOCK_SIZE 的 tile。
   2) 每个 block 负责计算 C 中一个 tile（BLOCK_M × BLOCK_N）。
   3) 沿 K 方向以 BLOCK_K 分块：
        · 每轮从 A、B 中提取一个 BLOCK_M×BLOCK_K / BLOCK_K×BLOCK_N 的子块，
          搬运到 shared memory。
        · 在 shared memory 中，对当前 K 范围内的数据做乘加，累加到寄存器 result。
   4) 遍历完所有 K-block 后，将 result 写回对应的 C 元素位置（融合 alpha / beta）。

4. 当前实现的限制与假设：
   - 假设：
       · BLOCK_M = BLOCK_N = BLOCK_K = BLOCK_SIZE
       · M、N、K 都能被 BLOCK_SIZE 整除；
   - 原因：
       · kernel 内没有做 gx/gy 越界检查；
       · 主函数中仅测试 N=1024（恰好是 32 的整数倍）。
   - 若要支持一般尺寸，需要在 kernel 内增加：
       if (global_row >= M || global_col >= N) return; 之类的边界判断。
------------------------------------------------------------
*/
template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // ------------------------------
    // 1. Block 在 C 中的 tile 坐标
    // ------------------------------
    int block_row = blockIdx.y; // 当前 block 负责的 C-tile 的“行 tile 索引”
    int block_col = blockIdx.x; // 当前 block 负责的 C-tile 的“列 tile 索引”

    // ------------------------------
    // 2. Tile / 子块尺寸定义（本例使用正方形 tile）
    // ------------------------------
    const int BLOCK_M = BLOCK_SIZE; // 一个 C-tile 的行数
    const int BLOCK_N = BLOCK_SIZE; // 一个 C-tile 的列数
    const int BLOCK_K = BLOCK_SIZE; // 每次从 K 方向加载的“厚度”（K 子块长度）

    // ------------------------------
    // 3. Thread 在 tile 内的局部坐标
    // ------------------------------
    int thread_row = threadIdx.y; // 线程在 tile 内的行索引 [0, BLOCK_M)
    int thread_col = threadIdx.x; // 线程在 tile 内的列索引 [0, BLOCK_N)

    // ------------------------------
    // 4. Block 级共享内存：缓存当前 K-block 的 A / B 子块
    //
    // 布局约定（row-major）：
    //   shared_A 视为 [BLOCK_M][BLOCK_K]：
    //       shared_A[r * BLOCK_K + c] = A_sub(r, c)
    //   shared_B 视为 [BLOCK_K][BLOCK_N]：
    //       shared_B[r * BLOCK_N + c] = B_sub(r, c)
    // ------------------------------
    __shared__ float shared_A[BLOCK_M * BLOCK_K];
    __shared__ float shared_B[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 对应到全局内存中的起始指针
    // ------------------------------
    // A_start：
    //   - 指向当前 block 负责的 A 行块的左上角（row = block_row * BLOCK_M, col = 0）
    //   - 后续通过 A_start += BLOCK_K 在 K 方向滑动子块
    const float* A_start = &A[block_row * BLOCK_M * K];

    // B_start：
    //   - 指向当前 block 负责的 B 列块的左上角（row = 0, col = block_col * BLOCK_N）
    //   - 后续通过 B_start += BLOCK_K * N 在 K 方向滑动子块
    const float* B_start = &B[block_col * BLOCK_N];

    // C_start：
    //   - 指向当前 block 负责的 C-tile 左上角：
    //       row = block_row * BLOCK_M
    //       col = block_col * BLOCK_N
    float* C_start = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ------------------------------
    // 6. 每个线程负责 C-tile 中一个元素的累加
    // ------------------------------
    float result = 0.0f;

    // ------------------------------
    // 7. 沿 K 方向分块累加
    //    外层循环：k_block = 0, BLOCK_K, 2*BLOCK_K, ...
    //    对应的 A_sub / B_sub：
    //       A_sub 行范围：block_row * BLOCK_M .. +BLOCK_M-1
    //       A_sub 列范围：k_block .. k_block+BLOCK_K-1
    //       B_sub 行范围：k_block .. k_block+BLOCK_K-1
    //       B_sub 列范围：block_col * BLOCK_N .. +BLOCK_N-1
    // ------------------------------
    for (int k_block = 0; k_block < K; k_block += BLOCK_K)
    {
        // --------------------------
        // 7.1 将当前 K-block 的 A、B 子块搬到共享内存
        //
        // 约定：BLOCK_M == blockDim.y，BLOCK_N == blockDim.x，BLOCK_K == BLOCK_SIZE
        //       每个线程加载一个元素：
        //
        // A：
        //   全局坐标：
        //     row = block_row * BLOCK_M + thread_row
        //     col = k_block + thread_col
        //   使用 A_start 和列偏移 BLOCK_K 累加的方式实现：
        //     A_start 已经指向 (block_row * BLOCK_M, k_block) 行块的左上角，
        //     这里直接按 row-major：
        //       A_start[thread_row * K + thread_col] == A(row, col)
        //
        // B：
        //   全局坐标：
        //     row = k_block + thread_row
        //     col = block_col * BLOCK_N + thread_col
        //   同理 B_start 已指向 (0, block_col * BLOCK_N)，再通过 B_start += BLOCK_K*N
        //   滑动行块：
        //       B_start[thread_row * N + thread_col] == B(row, col)
        // --------------------------
        shared_A[thread_row * BLOCK_K + thread_col] = A_start[thread_row * K + thread_col];
        shared_B[thread_row * BLOCK_N + thread_col] = B_start[thread_row * N + thread_col];

        // 确保共享内存中的 A、B 当前子块已被所有线程写入完毕
        __syncthreads();

        // --------------------------
        // 7.2 在共享内存中做本 K-block 的部分矩阵乘法
        //
        // 对于当前线程 (thread_row, thread_col)：
        //   result += Σ_{k_inner=0..BLOCK_K-1}
        //              shared_A[thread_row, k_inner] * shared_B[k_inner, thread_col]
        //
        // 线性下标：
        //   shared_A[thread_row * BLOCK_K + k_inner]
        //   shared_B[k_inner * BLOCK_N + thread_col]
        // --------------------------
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            result +=
                shared_A[thread_row * BLOCK_K + k_inner] * shared_B[k_inner * BLOCK_N + thread_col];
        }

        // 当前 K-block 计算结束，所有线程同步后再加载下一块
        __syncthreads();

        // --------------------------
        // 7.3 移动 A_start / B_start 到下一个 K-block：
        //
        // A_start：
        //   - 原始 A_start 指向 (block_row * BLOCK_M, k_block)
        //   - A 是 row-major，行跨度为 K
        //   - 每轮结束后 A_start += BLOCK_K：
        //       等价于列偏移 +BLOCK_K，即起始列从 k_block → k_block + BLOCK_K
        //
        // B_start：
        //   - 原始 B_start 指向 (0, block_col * BLOCK_N)
        //   - B 是 row-major，行跨度为 N
        //   - 每轮结束后 B_start += BLOCK_K * N：
        //       等价于行偏移 +BLOCK_K，即起始行从 k_block → k_block + BLOCK_K
        // --------------------------
        A_start += BLOCK_K;     // A 起始列整体 +BLOCK_K
        B_start += BLOCK_K * N; // B 起始行整体 +BLOCK_K
    }

    // ------------------------------
    // 8. 将累加结果写回 C（融合 alpha / beta）
    // ------------------------------
    // 该线程负责的 C 元素全局坐标：
    //   row = block_row * BLOCK_M + thread_row
    //   col = block_col * BLOCK_N + thread_col
    //
    // C_start 已经指向 (row = block_row * BLOCK_M, col = block_col * BLOCK_N)：
    //   C(row, col) = C_start[thread_row * N + thread_col]
    // ------------------------------
    C_start[thread_row * N + thread_col] =
        alpha * result + beta * C_start[thread_row * N + thread_col];
}

// ==============================
// 向上取整除法：用于计算 grid 尺寸
// ------------------------------
// 返回：ceil(M / N)，即至少需要多少个尺寸为 N 的块覆盖 M。
// ==============================
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main()
{
    // ==============================
    // 1. 测试矩阵大小（本例只测一个 1024×1024 方阵）
    // ==============================
    std::vector<int> sizes = {1024};

    // 打开 CSV：用于记录性能与正确性
    // 字段含义：
    //   Size           ：矩阵边长 N
    //   CUBLAS_GFLOPS  ：cuBLAS 实测 GFLOPS
    //   MySGEMM_FLOPS  ：mysgemm_v1 实测 GFLOPS
    //   Matched        ：1=与 cuBLAS 在 TOL 内一致，0=出现误差> TOL
    std::ofstream csv_file("sgemm_benchmark_v1.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        // A/B/C 都是 N×N，row-major
        size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

        // ==============================
        // 2. Host 端内存分配
        // ==============================
        float* host_a = (float*)malloc(size);
        float* host_b = (float*)malloc(size);
        float* host_c_cublas = (float*)malloc(size);
        float* host_c_v1 = (float*)malloc(size);

        // ==============================
        // 3. Device 端内存分配
        // ==============================
        float *device_a, *device_b, *device_c_v1;
        checkCudaError(cudaMalloc(&device_a, size), "cuda malloc device_a error");
        checkCudaError(cudaMalloc(&device_b, size), "cuda malloc device_b error");
        checkCudaError(cudaMalloc(&device_c_v1, size), "cuda malloc device_c_v1 error");

        bool out_of_memory = false;

        try
        {
            // ==============================
            // 4. 初始化 Host 端矩阵数据
            //    A 全 1，B 全 2
            //    对应理论结果：C_ij = Σ(1 * 2) = 2 * N
            // ==============================
            for (int i = 0; i < N * N; i++)
            {
                host_a[i] = 1.0f;
                host_b[i] = 2.0f;
            }

            // 拷贝 Host -> Device
            checkCudaError(cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_a to device error");
            checkCudaError(cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_b to device error");

            // ==============================
            // 5. 创建 cuBLAS 句柄
            // ==============================
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate error");

            float alpha = 1.0f, beta = 0.0f;

            // ==============================
            // 6. CUDA 事件：用于计时
            // ==============================
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start error");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop error");

            // ==============================
            // 7. cuBLAS SGEMM 预热
            //
            // 重要说明（列主序 vs 行主序）：
            //   - cuBLAS 假定输入为 column-major；
            //   - 我们的 device_a/device_b 实际是 row-major；
            //   - 这里通过“交换 A/B 顺序 + 方阵 + 相同布局比较”的方式，
            //     让 host 看到的结果在 row-major 空间中与手写核一致。
            //
            // 换句话说，这里不强调严格 BLAS 语义，而是：
            //   “在同一内存布局下，让 cuBLAS 和手写核计算的是同一个线性代数对象”，
            //   用于：
            //     1) 结果一致性验证
            //     2) 性能基线对比
            // ==============================
            int warmup_time = 10;
            for (int i = 0; i < warmup_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_b, N, // 左矩阵（列主序视角）
                                             device_a, N, // 右矩阵
                                             &beta, device_c_v1, N),
                                 "cublasSgemm error");
            }
            cudaDeviceSynchronize();

            // ==============================
            // 8. cuBLAS Benchmark 计时
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

            float cublas_time = 0.0f; // 毫秒
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime error");

            // 拷回 cuBLAS 结果
            checkCudaError(cudaMemcpy(host_c_cublas, device_c_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v1 to host_c_cublas error");

            // ==============================
            // 9. 手写 SGEMM 预热
            // ==============================
            checkCudaError(cudaMemset(device_c_v1, 0, size), "cudaMemset device_c_v1 error");

            dim3 blockDim(32, 32);                          // 每个 block 32×32 个线程
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32)); // 按 tile 覆盖整个 C

            for (int i = 0; i < warmup_time; i++)
            {
                mysgemm_v1<32>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v1);
            }
            cudaDeviceSynchronize();

            // ==============================
            // 10. 手写 SGEMM Benchmark 计时
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

            float v1_time = 0.0f; // 毫秒
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime error");

            // 拷回手写 SGEMM 结果
            checkCudaError(cudaMemcpy(host_c_v1, device_c_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v1 to host_c_v1 error");

            // ==============================
            // 11. 精度验证：最多报 10 个错误
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
            // 12. 计算 GFLOPS
            //    单次 GEMM ≈ 2 * N^3 FLOPs
            //    总 FLOPs = repeat_time * 2 * N^3
            //    时间单位是 ms，故除以 (time_ms * 1e6) 得到 GFLOPS
            // ==============================
            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);

            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // ==============================
            // 13. 释放资源
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

        // OOM 情况写入 CSV，方便后续分析
        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v1.csv'" << std::endl;
    return 0;
}
