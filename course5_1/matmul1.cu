#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf：用于浮点数误差比较
#include <fstream> // std::ofstream：用于写 CSV 文件
#include <iostream>
#include <vector>

#define TOL 1e-5f // 结果校验时允许的最大浮点误差

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
// cuBLAS 错误检查函数
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
mysgemm_v1: 使用共享内存 tiling 的手写 SGEMM Kernel
计算公式: C = alpha * A * B + beta * C

矩阵维度（row-major）:
    A: M × K
    B: K × N
    C: M × N

参数说明：
    M: 矩阵 A 的行数
    N: 矩阵 B / C 的列数
    K: 矩阵 A 的列数 = 矩阵 B 的行数
    alpha: 矩阵乘法缩放系数
    beta:  累加系数（控制是否叠加原有 C）
    A, B, C: 输入输出矩阵指针（row-major）

算法思路（典型 Block-level tiling）：
    1. 将 C 分块为 BLOCK_SIZE × BLOCK_SIZE 的 tile。
    2. 每个 block 负责计算 C 中一个 tile（BLOCK_M × BLOCK_N）。
    3. 沿 K 方向分块（BLOCK_K），每次将 A、B 对应 sub-tile 加载到共享内存。
    4. 在共享内存中完成这段 K 范围内的部分乘加，并累加到寄存器 result 中。
    5. 遍历完所有 K-block 后，写回 C 对应位置（融合 alpha / beta）。
    6. 当前实现假定 M、N、K 都能被 BLOCK_SIZE 整除（未做边界检查）。
*/
template <const int BLOCK_SIZE>
__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta,
                           float* C)
{
    // ==============================
    // Block 在 C 中的 tile 坐标
    // ==============================
    int block_row = blockIdx.y; // 当前 block 负责的 C-tile 的“行 tile 索引”
    int block_col = blockIdx.x; // 当前 block 负责的 C-tile 的“列 tile 索引”

    // ==============================
    // Tile 尺寸定义（本例使用正方形 tile）
    // ==============================
    const int BLOCK_M = BLOCK_SIZE; // C-tile 的行数
    const int BLOCK_N = BLOCK_SIZE; // C-tile 的列数
    const int BLOCK_K = BLOCK_SIZE; // 每次从 K 方向加载的“厚度”

    // ==============================
    // Thread 在 tile 内的坐标
    // ==============================
    int thread_row = threadIdx.y; // 线程在 tile 内的行索引 [0, BLOCK_M)
    int thread_col = threadIdx.x; // 线程在 tile 内的列索引 [0, BLOCK_N)

    // ==============================
    // 共享内存：缓存当前 K-block 对应的 A、B 子块
    // 形状：
    //   shared_A: BLOCK_M × BLOCK_K
    //   shared_B: BLOCK_K × BLOCK_N
    // ==============================
    __shared__ float shared_A[BLOCK_M * BLOCK_K];
    __shared__ float shared_B[BLOCK_K * BLOCK_N];

    // ==============================
    // 对应到全局内存中的起始指针
    //   A_start 指向当前 block 负责的 A 的首行、K=0 处
    //   B_start 指向当前 block 负责的 B 的首列、K=0 处
    //   C_start 指向当前 block 负责的 C-tile 左上角
    // ==============================
    const float* A_start = &A[block_row * BLOCK_M * K];
    const float* B_start = &B[block_col * BLOCK_N];
    float* C_start = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ==============================
    // 每个线程负责 C-tile 中一个元素的累加
    // ==============================
    float result = 0.0f;

    // ==============================
    // 沿 K 方向分块累加
    // 外层循环：k_block=0, BLOCK_K, 2*BLOCK_K, ...
    // ==============================
    for (int k_block = 0; k_block < K; k_block += BLOCK_K)
    {
        // ------------------------------
        // 1. 将当前 K-block 的 A、B 子块搬到共享内存
        //
        // A_sub(block_row*BLOCK_M + r, k_block + c) ->
        //   shared_A[r, c]  (r = thread_row, c = thread_col)
        //
        // B_sub(k_block + r, block_col*BLOCK_N + c) ->
        //   shared_B[r, c]  (r = thread_row, c = thread_col)
        //
        // 注意：当前实现假定 BLOCK_M/BLOCK_N/BLOCK_K 不超过 blockDim.y/x 等尺寸，
        //       每个线程加载一个元素。
        // ------------------------------
        shared_A[thread_row * BLOCK_K + thread_col] = A_start[thread_row * K + thread_col];
        shared_B[thread_row * BLOCK_N + thread_col] = B_start[thread_row * N + thread_col];

        // 确保 A、B 的当前 tile 已全部加载到共享内存
        __syncthreads();

        // ------------------------------
        // 2. 在共享内存中做本 K-block 的部分矩阵乘法
        //    对于当前线程 (thread_row, thread_col)：
        //    result += Σ shared_A[thread_row, k] * shared_B[k, thread_col]
        // ------------------------------
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            result +=
                shared_A[thread_row * BLOCK_K + k_inner] * shared_B[k_inner * BLOCK_N + thread_col];
        }

        // 计算完当前 K-block 后，所有线程同步，再加载下一块
        __syncthreads();

        // ------------------------------
        // 3. 移动 A_start / B_start 到下一个 K-block：
        //
        //   A_start 沿 K 方向前进 BLOCK_K 个元素（列偏移）
        //   B_start 沿 K 方向前进 BLOCK_K 行（每行有 N 个元素）
        // ------------------------------
        A_start += BLOCK_K;     // 等价于 A 的列索引整体 +BLOCK_K
        B_start += BLOCK_K * N; // 等价于 B 的行索引整体 +BLOCK_K
    }

    // ==============================
    // 写回结果到全局内存
    //
    // C 中该元素的行列为：
    //   row = block_row * BLOCK_M + thread_row
    //   col = block_col * BLOCK_N + thread_col
    // 这里 C_start 已经指向 (row, block_col*BLOCK_N) 的起点，
    // 所以只需加 thread_row*N + thread_col
    // ==============================
    C_start[thread_row * N + thread_col] =
        alpha * result + beta * C_start[thread_row * N + thread_col];
}

// ==============================
// 向上取整除法：用于计算 grid 尺寸
// ==============================
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

int main()
{
    // ==============================
    // 测试矩阵大小（本例只测一个 1024x1024 方阵）
    // ==============================
    std::vector<int> sizes = {1024};

    std::ofstream csv_file("sgemm_benchmark_v1.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        // A/B/C 都是 N×N，row-major
        size_t size = N * N * sizeof(float);

        // ==============================
        // Host 端内存分配
        // ==============================
        float* host_a = (float*)malloc(size);
        float* host_b = (float*)malloc(size);
        float* host_c_cublas = (float*)malloc(size);
        float* host_c_v1 = (float*)malloc(size);

        // ==============================
        // Device 端内存分配
        // ==============================
        float *device_a, *device_b, *device_c_v1;
        checkCudaError(cudaMalloc(&device_a, size), "cuda malloc device_a error");
        checkCudaError(cudaMalloc(&device_b, size), "cuda malloc device_b error");
        checkCudaError(cudaMalloc(&device_c_v1, size), "cuda malloc device_c_v1 error");

        bool out_of_memory = false;

        try
        {
            // ==============================
            // 初始化 Host 端矩阵数据
            //   A 全 1，B 全 2
            //   对应理论结果：C_ij = Σ(1 * 2) = 2 * N
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
            // 创建 cuBLAS 句柄
            // ==============================
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate error");

            float alpha = 1.0f, beta = 0.0f;

            // ==============================
            // CUDA 事件：用于计时
            // ==============================
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start error");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop error");

            // ==============================
            // cuBLAS SGEMM 预热
            //
            // 注意：cuBLAS 默认按列主序解释矩阵，
            // 这里以内存块视角调用（B、A 交换位置）+ 同一布局下的对比，
            // 只做“相对性能”和“数值一致性”验证，不做绝对 BLAS 语义校验。
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
            // cuBLAS Benchmark 计时
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
            // 手写 SGEMM 预热
            // ==============================
            checkCudaError(cudaMemset(device_c_v1, 0, size), "cudaMemset device_c_v1 error");

            dim3 blockDim(32, 32);                          // 每个 block 32×32 个线程
            dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(N, 32)); // 覆盖整个 C 的 tile 网格

            for (int i = 0; i < warmup_time; i++)
            {
                mysgemm_v1<32>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v1);
            }
            cudaDeviceSynchronize();

            // ==============================
            // 手写 SGEMM Benchmark 计时
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
            // 精度验证：最多报 10 个错误
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
            //   单次 GEMM ≈ 2 * N^3 FLOPs
            //   总 FLOPs = repeat_time * 2 * N^3
            //   时间单位是 ms，故除以 (time * 1e6) 得到 GFLOPS
            // ==============================
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

        // OOM 情况写入 CSV 方便排查
        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v1.csv'" << std::endl;
    return 0;
}
