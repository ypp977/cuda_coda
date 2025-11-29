#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf()：用于浮点数绝对值/误差比较
#include <fstream> // std::ofstream：写 CSV 文件
#include <iostream>
#include <vector>

#define TOL 1e-5f // 结果校验时允许的最大浮点误差

// ============================================================
// CUDA 错误检查工具函数
// ------------------------------------------------------------
// 用法：包装所有 cudaXXX 调用；
// 一旦返回值不是 cudaSuccess，打印提示信息和 CUDA 错误字符串并退出。
// ============================================================
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================
// cuBLAS 错误检查工具函数
// ------------------------------------------------------------
// 用法：包装所有 cublasXXX 调用；
// 一旦返回值不是 CUBLAS_STATUS_SUCCESS，打印提示信息和状态码并退出。
// ============================================================
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
模块名：mysgemm_v2（Block × Thread 两级 tiling 的 SGEMM kernel）
------------------------------------------------------------
1. 功能：
   - 实现单精度矩阵乘法：
       C = alpha * A * B + beta * C
   - 采用 Block-level + Thread-level 两级 tiling：
       · Block 负责一个 BLOCK_M × BLOCK_N 的 C 子块（C_tile）
       · 每个线程负责其中一个 THREAD_M × THREAD_N 的小子块

2. 矩阵维度与存储布局：
   - 约定为 row-major：
       A: M × K（行跨度 = K）
       B: K × N（行跨度 = N）
       C: M × N（行跨度 = N）

3. 参数说明：
   - M          : A 的行数（也是 C 的行数）
   - N          : B/C 的列数
   - K          : A 的列数 = B 的行数
   - alpha      : 矩阵乘法缩放系数
   - beta       : 累加系数，控制是否叠加原有 C
   - A, B, C    : 输入/输出矩阵指针（row-major）

4. 模板参数（几何含义）：
   - BLOCK_M, BLOCK_N：
       Block 级 C_tile 的高和宽，单位为“元素行/列”
   - BLOCK_K：
       沿 K 方向的分块厚度，每次迭代处理 BLOCK_K 个 K 元素
   - THREAD_M, THREAD_N：
       单个线程负责的 C 子块高度/宽度（Thread-level tile）

5. 算法结构：
   1) 将 C 按 BLOCK_M × BLOCK_N 划分为二维 tile 网格，每个 block 负责一个 tile。
   2) 在一个 block 内：
        · 逻辑上再划分为 (BLOCK_M / THREAD_M) × (BLOCK_N / THREAD_N) 个线程级子块；
        · 使用线性 threadIdx.x 将线程映射到 (thread_tile_row, thread_tile_col)。
   3) 沿 K 方向以 BLOCK_K 为步长循环：
        a) 所有线程协同，将当前 K 子块对应的 A/B 子矩阵加载到 shared memory：
             - shared_a: BLOCK_M × BLOCK_K
             - shared_b: BLOCK_K × BLOCK_N
        b) 每个线程从 shared_a/shared_b 中按自己的线程级子块位置取数据，
           在寄存器 tmp[THREAD_M][THREAD_N] 中完成乘加累积。
   4) 所有 K 子块累加完成后，将 tmp 中的结果按 alpha / beta 写回到 C。

6. 重要约束（由调用端保证，否则行为未定义）：
   - BLOCK_M % THREAD_M == 0，BLOCK_N % THREAD_N == 0
   - threads_per_block_x = BLOCK_N / THREAD_N
     threads_per_block_y = BLOCK_M / THREAD_M
     total_threads        = threads_per_block_x * threads_per_block_y
     且 total_threads == blockDim.x
   - 当前实现未做边界检查，假设：
       · M 是 BLOCK_M 的整数倍
       · N 是 BLOCK_N 的整数倍
       · K 是 BLOCK_K 的整数倍
     否则需要在写回 C 和共享内存加载处增加显式边界判断。
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C)
{
    // ------------------------------
    // 1. 当前 block 在 C 中的 tile 坐标
    //    block_row / block_col 决定该 block 负责哪一个 C_tile
    // ------------------------------
    int block_col = blockIdx.x; // C 在列方向上的 tile 索引
    int block_row = blockIdx.y; // C 在行方向上的 tile 索引

    // ------------------------------
    // 2. 计算 tile 内线程布局
    //    threads_per_block_x：N 方向上有多少个线程级子块
    //    threads_per_block_y：M 方向上有多少个线程级子块
    //    total_threads      ：该 block 内线程总数（必须 == blockDim.x）
    // ------------------------------
    int threads_per_block_x = BLOCK_N / THREAD_N;
    int threads_per_block_y = BLOCK_M / THREAD_M;
    int total_threads = threads_per_block_x * threads_per_block_y;

    // ------------------------------
    // 3. 当前线程在 C_tile 内负责的“线程级子块”的左上角局部坐标
    //    local_row / local_col：相对于当前 C_tile 的行/列偏移
    //    映射规则：
    //      - 先在 N 方向铺满一行，再在 M 方向换行（row-major 线程映射）
    // ------------------------------
    int local_col = (threadIdx.x % threads_per_block_x) * THREAD_N;
    int local_row = (threadIdx.x / threads_per_block_x) * THREAD_M;

    // ------------------------------
    // 4. Block 级共享内存：缓存当前 K-block 的 A / B 子块
    //
    //   shared_a：逻辑形状 [BLOCK_M][BLOCK_K]
    //             行跨度 = BLOCK_K
    //   shared_b：逻辑形状 [BLOCK_K][BLOCK_N]
    //             行跨度 = BLOCK_N
    //
    //   访问形式：
    //     shared_a[row * BLOCK_K + col]
    //     shared_b[row * BLOCK_N + col]
    // ------------------------------
    __shared__ float shared_a[BLOCK_M * BLOCK_K];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算当前 block 对应的全局矩阵基指针
    //
    //   A 基指针：
    //     指向 A(block_row * BLOCK_M, 0)
    //     即该 block 对应的 C 行块的起始行，K 维从 0 开始。
    //
    //   B 基指针：
    //     指向 B(0, block_col * BLOCK_N)
    //     即该 block 对应的 C 列块的起始列，K 维从 0 开始。
    //
    //   C 基指针：
    //     指向 C(block_row * BLOCK_M, block_col * BLOCK_N)
    // ------------------------------
    A = &A[block_row * BLOCK_M * K];
    B = &B[block_col * BLOCK_N];
    C = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ------------------------------
    // 6. 为加载 shared_a / shared_b 设计线程分工
    //
    //   对 A：
    //     a_load_row：该线程负责的 A 子块行索引（相对于 A_tile 内部）
    //     a_load_col：该线程负责的 A 子块列索引（相对于 A_tile 内部）
    //     a_load_stride：在 M 方向上的步长，让所有线程覆盖 BLOCK_M 行。
    //
    //   对 B：
    //     b_load_row：该线程负责的 B 子块行索引（相对于 B_tile 内部）
    //     b_load_col：该线程负责的 B 子块列索引（相对于 B_tile 内部）
    //     b_load_stride：在 K 方向上的步长，让所有线程覆盖 BLOCK_K 行。
    //
    //   要求：
    //     total_threads 能够整除 BLOCK_M * BLOCK_K 和 BLOCK_K * BLOCK_N，
    //     保证 for 循环内 i += stride 可以完整覆盖子块。
    // ------------------------------
    int a_load_row = threadIdx.x / BLOCK_K;
    int a_load_col = threadIdx.x % BLOCK_K;
    int a_load_stride = total_threads / BLOCK_K; // 一轮覆盖的行数（M 方向）

    int b_load_row = threadIdx.x / BLOCK_N;
    int b_load_col = threadIdx.x % BLOCK_N;
    int b_load_stride = total_threads / BLOCK_N; // 一轮覆盖的行数（K 方向）

    // ------------------------------
    // 7. 寄存器缓存：每个线程负责的 C 子块的累加结果
    //    尺寸为 THREAD_M × THREAD_N，初始为 0
    // ------------------------------
    float tmp[THREAD_M][THREAD_N] = {0.0f};

    // ------------------------------
    // 8. 沿 K 方向分块累加
    //    概念上：k_block = 0, BLOCK_K, 2*BLOCK_K, ...
    //    实现上：通过循环次数 + 指针偏移来隐含 k_block，下标里不直接用 k。
    // ------------------------------
#pragma unroll
    for (int k = 0; k < K; k += BLOCK_K)
    {
        // --------------------------
        // 8.1 从全局内存加载当前 K-block 的 A 子块到 shared_a
        //
        //   逻辑上的 A_tile 当前子块：
        //     行：block_row * BLOCK_M .. block_row * BLOCK_M + BLOCK_M - 1
        //     列：k .. k + BLOCK_K - 1
        //
        //   由于 A 指针在每一轮循环结束后都会 += BLOCK_K，
        //   此处 A[(a_load_row + i) * K + a_load_col]
        //   实际等价于访问当前 K 子块的列范围。
        //
        //   循环 i 以 a_load_stride 为步长，在 M 方向（行）上展开，
        //   确保所有线程共同将 BLOCK_M × BLOCK_K 填满 shared_a。
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            shared_a[(a_load_row + i) * BLOCK_K + a_load_col] =
                A[(a_load_row + i) * K + a_load_col];
        }

        // --------------------------
        // 8.2 从全局内存加载当前 K-block 的 B 子块到 shared_b
        //
        //   逻辑上的 B_tile 当前子块：
        //     行：k .. k + BLOCK_K - 1
        //     列：block_col * BLOCK_N .. block_col * BLOCK_N + BLOCK_N - 1
        //
        //   同样地，B 指针在每轮循环结束后都会 += BLOCK_K * N，
        //   使得 B[(b_load_row + i) * N + b_load_col]
        //   始终位于当前 K 子块的行范围内。
        //
        //   循环 i 以 b_load_stride 为步长，在 K 方向（行）上展开，
        //   确保所有线程共同将 BLOCK_K × BLOCK_N 填满 shared_b。
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            shared_b[(b_load_row + i) * BLOCK_N + b_load_col] =
                B[(b_load_row + i) * N + b_load_col];
        }

        // 等待所有线程完成共享内存加载
        __syncthreads();

        // --------------------------
        // 8.3 预先更新 A、B 的全局指针到“下一 K-block”
        //
        //   本轮之后的计算完全来自 shared_a / shared_b，
        //   因此这里可以提前增加 A/B 指针，指向下一轮要加载的 K 子块。
        //
        //   A += BLOCK_K：
        //     按列偏移 BLOCK_K 列（row-major 下为线性偏移 BLOCK_K）
        //
        //   B += BLOCK_K * N：
        //     按行偏移 BLOCK_K 行，每行 N 个元素。
        // --------------------------
        A += BLOCK_K;
        B += BLOCK_K * N;

        // --------------------------
        // 8.4 使用共享内存中的 A/B 子块进行乘加累积
        //
        //   对于当前线程负责的局部 C 子块：
        //     (local_row + j, local_col + l)，其中
        //       j ∈ [0, THREAD_M)，l ∈ [0, THREAD_N)
        //
        //   计算：
        //     tmp[j][l] += Σ_{i=0..BLOCK_K-1}
        //        shared_a[(local_row + j), i] * shared_b[i, (local_col + l)]
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_K; i++)
        {
#pragma unroll
            for (int j = 0; j < THREAD_M; j++)
            {
#pragma unroll
                for (int l = 0; l < THREAD_N; l++)
                {
                    tmp[j][l] += shared_a[(local_row + j) * BLOCK_K + i] *
                                 shared_b[i * BLOCK_N + (local_col + l)];
                }
            }
        }

        // 等待所有线程完成本轮计算，再进入下一轮 K-block 的加载
        __syncthreads();
    }

    // ------------------------------
    // 9. 将寄存器 tmp 中的结果写回 C（融合 alpha / beta）
    //
    //   全局坐标：
    //     row_global = block_row * BLOCK_M + local_row + j
    //     col_global = block_col * BLOCK_N + local_col + l
    //
    //   由于 C 指针已指向：
    //     C(block_row * BLOCK_M, block_col * BLOCK_N)
    //
    //   所以该元素在线性空间中的索引为：
    //     c_index = (local_row + j) * N + (local_col + l)
    // ------------------------------
#pragma unroll
    for (int j = 0; j < THREAD_M; j++)
    {
        for (int l = 0; l < THREAD_N; l++)
        {
            int c_index = (local_row + j) * N + (local_col + l);
            C[c_index] = alpha * tmp[j][l] + beta * C[c_index];
        }
    }
}

// 向上取整：用于根据 BLOCK/TILE 尺寸计算 grid 维度
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// 生成测试矩阵尺寸（如需多尺寸测试可在此扩展）
std::vector<int> generateSizes()
{
    return {4096}; // 当前示例仅测试单一尺寸：4096 × 4096 方阵
}

int main()
{
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");

    std::vector<int> sizes = generateSizes();

    // CSV 字段：
    //   Size           ：矩阵边长 N
    //   CUBLAS_GFLOPS  ：cuBLAS 实测 GFLOPS
    //   MySGEMM_FLOPS  ：mysgemm_v2 实测 GFLOPS
    //   Matched        ：1=与 cuBLAS 在 TOL 内一致，0=存在误差> TOL
    std::ofstream csv_file("sgemm_benchmark_v2.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float);

        // --------------------------
        // Host 端内存分配
        // --------------------------
        float* host_a = (float*)malloc(size);
        float* host_b = (float*)malloc(size);
        float* host_c_cublas = (float*)malloc(size);
        float* host_c_v2 = (float*)malloc(size);

        // --------------------------
        // Device 端内存分配
        // --------------------------
        float *device_a, *device_b, *device_c_v2;
        checkCudaError(cudaMalloc(&device_a, size), "cudaMalloc device_a failed");
        checkCudaError(cudaMalloc(&device_b, size), "cudaMalloc device_b failed");
        checkCudaError(cudaMalloc(&device_c_v2, size), "cudaMalloc device_c_v2 failed");

        bool out_of_memory = false;

        try
        {
            // --------------------------
            // 初始化 A、B：A 全 1，B 全 2
            // 理论上（以数学视角）：C_ij = Σ(1 * 2) = 2 * N
            // --------------------------
            for (int i = 0; i < N * N; i++)
            {
                host_a[i] = 1.0f;
                host_b[i] = 2.0f;
            }

            checkCudaError(cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy device_a failed");
            checkCudaError(cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy device_b failed");

            // --------------------------
            // 创建 cuBLAS handle
            // --------------------------
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            // --------------------------
            // 创建 CUDA 事件用于计时
            // --------------------------
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");

            int warmup_times = 10;
            int repeat_times = 50;

            // --------------------------
            // cuBLAS 预热
            //
            // 关于存储布局说明：
            //   - cuBLAS 语义上假定矩阵为 column-major；
            //   - 这里的 device_a/device_b 是 row-major N×N；
            //   - 本基准只关注“与手写 kernel 在同一内存布局下的数值一致性”和
            //     “相对性能对比”，不追求严格 BLAS 语义等价。
            //   - 在当前填充模式（A 全 1，B 全 2）下，无论是否转置，
            //     结果矩阵中每个元素都等于 2*N，因此可安全用作数值参考。
            // --------------------------
            for (int i = 0; i < warmup_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v2, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(device_c_v2, 0, size), "cudaMemset device_c_v2 failed");

            // --------------------------
            // cuBLAS 计时测试
            // --------------------------
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v2, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

            float cublas_time = 0.0f; // 毫秒
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime failed");

            checkCudaError(cudaMemcpy(host_c_cublas, device_c_v2, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_c_cublas failed");

            // 为自定义 kernel 准备：C 清零
            checkCudaError(cudaMemset(device_c_v2, 0, size), "cudaMemset device_c_v2 failed");

            // --------------------------
            // MySGEMM 配置与预热
            //
            // 选型示例：
            //   BLOCK_M = BLOCK_N = 128，BLOCK_K = 8
            //   THREAD_M = THREAD_N = 8：
            //     · 每个 block 覆盖 128×128 的 C_tile；
            //     · 每个线程输出 8×8 个元素；
            //     · threads_per_block_x = 128 / 8 = 16
            //       threads_per_block_y = 128 / 8 = 16
            //       total_threads        = 256 = blockDim.x
            //
            //   N = 4096 可以被 128 整除，且 K=4096 可以被 BLOCK_K=8 整除，
            //   故无需额外边界判断。
            // --------------------------
            dim3 blockDim(256); // 一维线程块，内部自行映射到 (row, col)
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warmup_times; i++)
            {
                mysgemm_v2<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v2);
            }
            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(device_c_v2, 0, size), "cudaMemset device_c_v2 failed");

            // --------------------------
            // MySGEMM 计时测试
            // --------------------------
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                mysgemm_v2<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v2);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

            float v2_time = 0.0f; // 毫秒
            checkCudaError(cudaEventElapsedTime(&v2_time, start, stop),
                           "cudaEventElapsedTime failed");

            checkCudaError(cudaMemcpy(host_c_v2, device_c_v2, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_c_v2 failed");

            // --------------------------
            // 结果校验：与 cuBLAS 对比，允许误差 TOL
            // 只统计前 10 个超过阈值的差异，避免刷屏
            // --------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_c_cublas[i] - host_c_v2[i]) > TOL)
                {
                    error_count++;
                }
            }

            // --------------------------
            // 计算 GFLOPS
            //   单次 GEMM ≈ 2 * N^3 FLOPs（乘法 + 加法）
            //   总 FLOPs = repeat_times * 2 * N^3
            //   时间单位为 ms，故除以 (time_ms * 1e6) → GFLOPS
            // --------------------------
            float cublas_gflops = repeat_times * 2.0f * N * N * N / (cublas_time * 1e6f);
            float v2_gflops = repeat_times * 2.0f * N * N * N / (v2_time * 1e6f);

            csv_file << N << "," << cublas_gflops << "," << v2_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // --------------------------
            // 释放资源
            // --------------------------
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_a);
            cudaFree(device_b);
            cudaFree(device_c_v2);

            free(host_a);
            free(host_b);
            free(host_c_cublas);
            free(host_c_v2);
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
            // OOM 或其它异常时写入占位信息，便于后处理
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v2.csv'" << std::endl;
    return 0;
}
