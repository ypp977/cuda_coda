#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf()：用于浮点数绝对值/误差比较
#include <fstream> // std::ofstream：写 CSV 文件
#include <iostream>
#include <vector>

#define TOL 1e-5f // 结果校验时允许的最大浮点误差

// 检查 CUDA API 调用是否出错
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 调用是否出错
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
mysgemm_v2: 分块矩阵乘法（Tiled SGEMM）
计算公式: C = alpha * A * B + beta * C
------------------------------------------

参数说明：
    M: 矩阵 A 的行数
    N: 矩阵 B 的列数（也是矩阵 C 的列数）
    K: 矩阵 A 的列数 = 矩阵 B 的行数
    alpha: 矩阵乘法缩放系数
    beta:  累加系数（控制是否叠加原有 C）
    A, B, C: 输入输出矩阵（约定为 row-major 存储）

矩阵维度：
    A: M × K
    B: K × N
    C: M × N

算法设计思路（Block/Thread 两级 tiling）：
    1. 将 C 按 BLOCK_M × BLOCK_N 分块，每个 block 负责一个 C_tile。
    2. 每个线程在该 tile 内负责一个 THREAD_M × THREAD_N 的小块（线程级 tile）。
    3. 使用共享内存缓存 A、B 在当前 K 子块上的子矩阵，减少全局内存访问。
    4. 沿 K 方向分块（BLOCK_K 为步长），在寄存器中累加对应的乘加结果到 tmp。
    5. 循环结束后，将 tmp 中的结果按 alpha/beta 写回 C。
    6. 当前实现假设 M、N、K 都是 BLOCK_M/BLOCK_N/BLOCK_K 的整数倍（未做越界保护）。
------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C)
{
    // ------------------------------
    // 1. 当前 block 在 C 中的 tile 坐标
    //    block_row / block_col 决定 C_tile 的左上角所在的 tile 索引
    // ------------------------------
    int block_col = blockIdx.x; // C 在列方向上的第 block_col 个 tile
    int block_row = blockIdx.y; // C 在行方向上的第 block_row 个 tile

    // ------------------------------
    // 2. 计算 tile 内线程布局
    //    threads_per_block_x：在 N 方向上有多少个线程负责不同的 THREAD_N 子块
    //    threads_per_block_y：在 M 方向上有多少个线程负责不同的 THREAD_M 子块
    //    total_threads      ：该 block 内线程总数（要求 == blockDim.x）
    // ------------------------------
    int threads_per_block_x = BLOCK_N / THREAD_N;
    int threads_per_block_y = BLOCK_M / THREAD_M;
    int total_threads = threads_per_block_x * threads_per_block_y;

    // ------------------------------
    // 3. 当前线程在 C_tile 内负责的“线程级子块”的左上角坐标
    //    local_row / local_col 相对于当前 C_tile 的局部坐标，而不是全局坐标
    //    threadIdx.x 在 [0, total_threads) 内被线性映射到 (row, col)
    // ------------------------------
    int local_col = (threadIdx.x % threads_per_block_x) * THREAD_N;
    int local_row = (threadIdx.x / threads_per_block_x) * THREAD_M;

    // ------------------------------
    // 4. 共享内存：缓存当前 K-block 对应的 A、B 子块
    //    shared_a 形状：BLOCK_M × BLOCK_K
    //    shared_b 形状：BLOCK_K × BLOCK_N
    // ------------------------------
    __shared__ float shared_a[BLOCK_M * BLOCK_K];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算当前 block 对应的全局矩阵基指针
    //
    //    A 基指针指向：
    //      行 = block_row * BLOCK_M，列 = 0
    //      即 A(block_row * BLOCK_M, 0)
    //
    //    B 基指针指向：
    //      行 = 0，列 = block_col * BLOCK_N
    //      即 B(0, block_col * BLOCK_N)
    //
    //    C 基指针指向：
    //      行 = block_row * BLOCK_M，列 = block_col * BLOCK_N
    //      即 C(block_row * BLOCK_M, block_col * BLOCK_N)
    // ------------------------------
    A = &A[block_row * BLOCK_M * K];
    B = &B[block_col * BLOCK_N];
    C = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ------------------------------
    // 6. 为加载共享内存中的 A/B 子块设计线程分工
    //
    //    a_load_row / a_load_col：该线程负责加载 shared_a 中哪一行哪一列的元素
    //    a_load_stride          ：在 M 方向上的跨步，用来让所有线程协同覆盖 BLOCK_M 行
    //
    //    b_load_row / b_load_col：该线程负责加载 shared_b 中哪一行哪一列的元素
    //    b_load_stride          ：在 K 方向上的跨步，用来让所有线程协同覆盖 BLOCK_K 行
    // ------------------------------
    int a_load_row = threadIdx.x / BLOCK_K;
    int a_load_col = threadIdx.x % BLOCK_K;
    int a_load_stride = total_threads / BLOCK_K; // 每次在 M 方向跨多少行

    int b_load_row = threadIdx.x / BLOCK_N;
    int b_load_col = threadIdx.x % BLOCK_N;
    int b_load_stride = total_threads / BLOCK_N; // 每次在 K 方向跨多少行

    // ------------------------------
    // 7. 寄存器缓存：每个线程负责的 C 子块的累加结果
    //    尺寸为 THREAD_M × THREAD_N
    // ------------------------------
    float tmp[THREAD_M][THREAD_N] = {0.0f};

    // ------------------------------
    // 8. 沿 K 方向分块累加
    //    每轮处理 BLOCK_K 宽度的 K 子块
    // ------------------------------
#pragma unroll
    for (int k = 0; k < K; k += BLOCK_K)
    {
        // --------------------------
        // 8.1 从全局内存加载 A 子块到 shared_a
        //     行范围：block_row * BLOCK_M ~ block_row * BLOCK_M + BLOCK_M - 1
        //     列范围：k ~ k + BLOCK_K - 1
        //
        //     这里通过 a_load_row + i 在 M 方向展开，
        //     a_load_stride 保证所有线程共同覆盖 BLOCK_M 行。
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            shared_a[(a_load_row + i) * BLOCK_K + a_load_col] =
                A[(a_load_row + i) * K + a_load_col];
        }

        // --------------------------
        // 8.2 从全局内存加载 B 子块到 shared_b
        //     行范围：k ~ k + BLOCK_K - 1
        //     列范围：block_col * BLOCK_N ~ block_col * BLOCK_N + BLOCK_N - 1
        //
        //     这里通过 b_load_row + i 在 K 方向展开，
        //     b_load_stride 保证所有线程共同覆盖 BLOCK_K 行。
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
        //     当前轮计算只访问 shared_a / shared_b，
        //     因此这里提前调整 A/B 指针不会影响本轮计算。
        // --------------------------
        A += BLOCK_K;     // A 沿列方向前进 BLOCK_K 列
        B += BLOCK_K * N; // B 沿行方向前进 BLOCK_K 行（每行跨度为 N）

        // --------------------------
        // 8.4 使用共享内存中的子块进行乘加累积
        //
        //     对于当前线程负责的局部坐标 (local_row + j, local_col + l)：
        //     tmp[j][l] += Σ shared_a[(local_row + j), i] * shared_b[i, (local_col + l)]
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

        // 等待所有线程完成本轮计算，再进入下一轮 K-block
        __syncthreads();
    }

    // ------------------------------
    // 9. 将寄存器 tmp 中的结果写回 C
    //
    //    全局坐标：
    //      row_global = block_row * BLOCK_M + local_row + j
    //      col_global = block_col * BLOCK_N + local_col + l
    //
    //    C 基指针已指向 (block_row * BLOCK_M, block_col * BLOCK_N)，
    //    因此索引为 row_offset * N + col_offset。
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

// 生成测试矩阵尺寸（可按需扩展多个尺寸）
std::vector<int> generateSizes()
{
    return {4096}; // 当前示例仅测试 4096×4096 方阵
}

int main()
{
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");

    std::vector<int> sizes = generateSizes();

    std::ofstream csv_file("sgemm_benchmark_v2.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;
        size_t size = (size_t)N * N * sizeof(float);

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
            // 理论上 C 中每个元素 ≈ 2 * N
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
            // cuBLAS 按列主序解释矩阵，但此处 A/B/C 为 N×N，
            // 且手写 kernel 与 cuBLAS 使用相同内存布局，
            // 只比较数值结果与性能，不做严格 BLAS 语义区分。
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
            // MySGEMM 预热
            //
            // BLOCK_M = BLOCK_N = 128，THREAD_M = THREAD_N = 8：
            //   每个 block 覆盖 128×128 的 C_tile，
            //   每个 block 含 256 个线程（dim3 blockDim(256)），
            //   每线程输出 8×8 个元素。
            // 由于 N=4096 可以被 128 整除，因此未做边界判断也是安全的。
            // --------------------------
            dim3 blockDim(256); // 一维线程块，但在线程内部自行映射到 (row,col)
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128)); // 每个维度按 128 大小划分 tile

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
            // 只统计前 10 个超过阈值的差异
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
            // 单次 GEMM ≈ 2 * N^3 FLOPs（乘加各算一次）
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
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v2.csv'" << std::endl;
    return 0;
}
