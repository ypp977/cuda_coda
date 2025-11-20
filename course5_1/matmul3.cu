#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf：用于比较浮点数误差
#include <fstream> // std::ofstream：用于写 CSV 文件
#include <iostream>
#include <vector>

// 结果对比时的误差容忍度
#define TOL 1e-5f
// 计算二维数组中元素的线性下标：row * leading_dimension + col
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 以 float4 形式访问连续 4 个 float（用于向量化读写）
// 注意：调用方需确保地址按 sizeof(float4) 对齐
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 检查 CUDA runtime API 调用是否出错
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 调用是否出错
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS Error: " << msg << " - Status code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------
mysgemm_v3: 高性能手写 SGEMM Kernel
计算公式: C = alpha * A * B + beta * C
------------------------------------------

模板参数：
    BLOCK_M, BLOCK_N, BLOCK_K : 每个 thread block 处理的 C 子块在 M/N/K 方向上的尺寸
    THREAD_M, THREAD_N        : 每个线程计算的 C 子块尺寸（行 × 列）

函数参数：
    M, N, K       : 矩阵维度 (A[M×K], B[K×N], C[M×N])
    alpha, beta   : 缩放系数（与 BLAS 约定一致）
    A, B          : 输入矩阵（约定为 row-major）
    C             : 输出矩阵（row-major）

算法设计思路：
    1. 将 C 拆分为 BLOCK_M × BLOCK_N 的 tile，每个 block 负责一个 tile。
    2. 每个线程在该 tile 内负责一个 THREAD_M × THREAD_N 的输出小块。
    3. 使用共享内存缓存 A、B 在当前 K-block 上的子矩阵，减少全局内存访问。
       - A 子块在共享内存中存为“转置布局”（[K][M]），便于后续按行连续访问。
       - B 子块在共享内存中保持 [K][N] 布局。
    4. 通过 float4 向量化读写，减少访问指令次数，提高带宽利用率。
    5. 沿 K 方向以 BLOCK_K 为步长分块，在寄存器里累加部分结果到 accum。
    6. 循环结束后，将 accum 中的结果按 alpha/beta 融合后写回 C。
    7. 当前实现假设：
       - BLOCK_M, BLOCK_N, BLOCK_K 能整除 M, N, K；
       - BLOCK_K 和 THREAD_N 为 4 的倍数，以满足 float4 对齐与边界要求；
       - blockDim.x == (BLOCK_M/THREAD_M) * (BLOCK_N/THREAD_N)。
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float* __restrict__ A,
                           float* __restrict__ B, float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    //    block_row_idx / block_col_idx 是以 tile 为单位的行列索引
    // ------------------------------
    const int block_col_idx = blockIdx.x;
    const int block_row_idx = blockIdx.y;

    // ------------------------------
    // 2. tile 内线程布局
    //    thread_per_row：沿 N 方向有多少个“线程级子块”
    //    thread_per_col：沿 M 方向有多少个“线程级子块”
    //    thread_per_block：该 block 中线程总数（需与 blockDim.x 一致）
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    // ------------------------------
    // 3. 当前线程在 C 的 tile 内负责的输出子块左上角坐标（局部坐标）
    //    local_row_idx / local_col_idx 是相对于本 tile 左上角的偏移
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M;

    // ------------------------------
    // 4. 共享内存：缓存 A/B 在当前 K-block 上的子块
    //
    //    shared_a：按 [K][M]（行主序，leading_dim = BLOCK_M）存储 A 的“转置子块”
    //              即 shared_a[k, m] = A_tile[m, k]
    //    shared_b：按 [K][N]（行主序，leading_dim = BLOCK_N）存储 B 子块
    //              即 shared_b[k, n] = B_tile[k, n]
    // ------------------------------
    __shared__ float shared_a[BLOCK_K * BLOCK_M];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算每个线程需要加载的 float4 向量数量
    //    总元素数 / 4 再平均分给每个线程
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 为加载 A/B 子块设计每个线程的起始位置与步长
    //
    //    对 A：
    //      a_load_row：该线程在 A 子块中的“行”起点（按 float 标量维度）
    //      a_load_col：该线程在 A 子块中的“列”起点（一次加载 4 个 float，故乘 4）
    //      a_load_stride：在 BLOCK_M 方向上的行步长，用于在 for 循环中覆盖所有行
    //
    //    对 B：
    //      b_load_row：该线程在 B 子块中的“行”起点
    //      b_load_col：该线程在 B 子块中的“列”起点（一次加载 4 个 float）
    //      b_load_stride：在 K 方向上的行步长，用于覆盖整个 BLOCK_K
    // ------------------------------
    const int a_load_row = threadIdx.x / (BLOCK_K / 4);
    const int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int a_load_stride = BLOCK_M / vec4_load_per_thread_a;

    const int b_load_row = threadIdx.x / (BLOCK_N / 4);
    const int b_load_col = (threadIdx.x % (BLOCK_N / 4)) * 4;
    const int b_load_stride = BLOCK_K / vec4_load_per_thread_b;

    // ------------------------------
    // 7. 寄存器缓存
    //
    //    accum      ：当前线程负责的 THREAD_M × THREAD_N C 子块的累加结果
    //    reg_a_vec  ：临时保存从全局内存读出的 A 的 float4 数据，用于写入 shared_a
    //    reg_a_tile ：当前 k_inner 下，A 的一列（或一组行）的标量值
    //    reg_b_tile ：当前 k_inner 下，B 的一行对应的 THREAD_N 个标量值
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.0f};
    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.0f};
    float reg_a_tile[THREAD_M];
    float reg_b_tile[THREAD_N];

    // ------------------------------
    // 8. 将 A/B/C 指针偏移到当前 block 对应的子矩阵起始位置（全局坐标）
    //
    //    A 指向：第 block_row_idx 个 tile 的第一行（行偏移为 block_row_idx * BLOCK_M）
    //    B 指向：第 block_col_idx 个 tile 的第一列（列偏移为 block_col_idx * BLOCK_N）
    //    C 指向：C 中对应的 tile 左上角元素
    // ------------------------------
    A = &A[block_row_idx * BLOCK_M * K];
    B = &B[block_col_idx * BLOCK_N];
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N];

    // ------------------------------
    // 9. 沿 K 方向按 BLOCK_K 分块循环
    // ------------------------------
#pragma unroll
    for (int k_block_start = 0; k_block_start < K; k_block_start += BLOCK_K)
    {
        // ---- 9.1 将 A 当前 K-block 子块从全局内存加载到 shared_a，并在写入时完成转置 ----
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            // reg_idx：当前循环中使用 reg_a_vec 的起始下标（以标量计）
            const int reg_idx = (i / a_load_stride) * 4;

            // 从全局 A 中按 float4 方式读取：
            //   读取位置为 A[a_load_row + i, a_load_col ... a_load_col+3]
            FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

            // 写入 shared_a 时进行“转置”：
            //   原：A 子块形状为 [BLOCK_M][BLOCK_K]
            //   现：shared_a 按 [BLOCK_K][BLOCK_M] 存储，便于后续以 k 为主维度访问
            shared_a[OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
            shared_a[OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
            shared_a[OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
            shared_a[OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
        }

        // ---- 9.2 将 B 当前 K-block 子块从全局内存加载到 shared_b（保持 [K][N] 布局） ----
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            FETCH_FLOAT4(shared_b[OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
        }

        // 等待所有线程完成 A/B 子块的共享内存加载
        __syncthreads();

        // ---- 9.3 将全局 A/B 指针移动到下一个 K-block 的起点
        //      当前循环只使用 shared_a/shared_b，移动不会影响本轮计算
        // ----
        A += BLOCK_K;     // A 沿列方向前进 BLOCK_K 列
        B += BLOCK_K * N; // B 沿行方向前进 BLOCK_K 行（每行有 N 个元素）

        // ---- 9.4 使用共享内存中的 A/B 子块进行乘加累积（K 内层循环） ----
#pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            // 从 shared_a 中取出当前 k_inner 对应的 A 列，映射到当前线程负责的 THREAD_M 行
            //   访问布局：shared_a[k_inner, local_row_idx + m]
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[m]) =
                    FETCH_FLOAT4(shared_a[OFFSET(k_inner, local_row_idx + m, BLOCK_M)]);
            }

            // 从 shared_b 中取出当前 k_inner 对应的 B 行，映射到当前线程负责的 THREAD_N 列
            //   访问布局：shared_b[k_inner, local_col_idx + n]
#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[n]) =
                    FETCH_FLOAT4(shared_b[OFFSET(k_inner, local_col_idx + n, BLOCK_N)]);
            }

            // 完成当前 k_inner 的 rank-1 更新：
            //   accum[m][n] += A_val(m) * B_val(n)
#pragma unroll
            for (int m = 0; m < THREAD_M; m++)
            {
#pragma unroll
                for (int n = 0; n < THREAD_N; n++)
                {
                    accum[m][n] += reg_a_tile[m] * reg_b_tile[n];
                }
            }
        }

        // 确保所有线程完成本轮 K-block 的累加，再进入下一轮
        __syncthreads();
    }

    // ------------------------------
    // 10. 将累加结果写回 C（按 float4 写回，融合 alpha/beta）
    //
    //     全局下标：
    //       row_global = block_row_idx * BLOCK_M + local_row_idx + m
    //       col_global = block_col_idx * BLOCK_N + local_col_idx + n
    //
    //     由于 C 指针已调整为 tile 左上角，故直接用 (local_row_idx + m, local_col_idx + n)
    //     作为相对坐标，并使用 N 作为行跨度。
    // ------------------------------
#pragma unroll
    for (int m = 0; m < THREAD_M; m++)
    {
#pragma unroll
        for (int n = 0; n < THREAD_N; n += 4)
        {
            float4 c_val = FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]);

            c_val.x = alpha * accum[m][n] + beta * c_val.x;
            c_val.y = alpha * accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * accum[m][n + 3] + beta * c_val.w;

            FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]) = c_val;
        }
    }
}

// 向上取整除法：用于根据 tile 尺寸计算 grid 维度
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// 生成测试矩阵大小（可按需扩展）
std::vector<int> generateSizes()
{
    return {4096};
}

// 主函数：对比 cuBLAS 与 mysgemm_v3 的性能和正确性
int main()
{
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "Failed to set CUDA device");

    std::vector<int> sizes = generateSizes();

    // 创建 CSV 文件记录测试结果
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // 遍历不同矩阵尺寸（当前只有一个 4096）
    for (int N : sizes)
    {
        std::cout << "Testing Size: " << N << std::endl;

        // A/B/C 都是 N×N，按 row-major 存储
        size_t size = static_cast<size_t>(N) * N * sizeof(float);

        // Host 端内存分配
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_v3 = (float*)malloc(size);

        // Device 端内存分配
        float *device_a, *device_b, *device_c_v3;
        checkCudaError(cudaMalloc(&device_a, size),
                       "Failed to allocate device memory for matrix A");
        checkCudaError(cudaMalloc(&device_b, size),
                       "Failed to allocate device memory for matrix B");
        checkCudaError(cudaMalloc(&device_c_v3, size),
                       "Failed to allocate device memory for matrix C");

        try
        {
            // 初始化 A、B：A 全 1，B 全 2
            // 理论上 C 的每个元素 ≈ 2 * N
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // 拷贝 A/B 到设备
            checkCudaError(cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice),
                           "Failed to copy matrix A from host to device");
            checkCudaError(cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice),
                           "Failed to copy matrix B from host to device");

            // 创建 cuBLAS handle
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

            float alpha = 1.0f, beta = 0.0f;

            // 创建 CUDA 事件用于计时
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "Failed to create CUDA start event");
            checkCudaError(cudaEventCreate(&stop), "Failed to create CUDA stop event");

            int warmup_time = 10;
            int repeat_time = 50;

            // --------------------------
            // cuBLAS 预热（避免首次调用带来的冷启动影响）
            // --------------------------
            for (int i = 0; i < warmup_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v3, N),
                                 "Failed to execute cuBLAS SGEMM operation");
            }
            cudaDeviceSynchronize();

            // --------------------------
            // cuBLAS 正式计时
            // --------------------------
            checkCudaError(cudaEventRecord(start), "Failed to record start event");
            for (int i = 0; i < repeat_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v3, N),
                                 "Failed to execute cuBLAS SGEMM operation");
            }
            checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
            checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

            float cublas_time = 0.0f; // ms
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "Failed to calculate elapsed time");

            // 拷回 cuBLAS 结果
            checkCudaError(cudaMemcpy(host_C_cublas, device_c_v3, size, cudaMemcpyDeviceToHost),
                           "Failed to copy result matrix from device to host");

            // 为自定义 SGEMM 准备：清零 C
            checkCudaError(cudaMemset(device_c_v3, 0, size),
                           "Failed to initialize matrix C with zeros");

            // --------------------------
            // 自定义 SGEMM 的执行配置
            //
            //   BLOCK_M = BLOCK_N = 128, THREAD_M = THREAD_N = 8：
            //     每个 block 负责 128×128 的 C tile；
            //     每个 block 有 256 个线程（blockDim.x = 256）；
            //     每个线程输出 8×8 个元素。
            //   N = 4096，可以被 128 整除，保证 tile 正好覆盖。
            // --------------------------
            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            // 自定义 SGEMM 预热
            for (int i = 0; i < warmup_time; i++)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v3);
            }
            cudaDeviceSynchronize();

            // 正式计时自定义 SGEMM
            checkCudaError(cudaEventRecord(start), "Failed to record start event for custom SGEMM");
            for (int i = 0; i < repeat_time; i++)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v3);
            }
            checkCudaError(cudaEventRecord(stop), "Failed to record stop event for custom SGEMM");
            checkCudaError(cudaEventSynchronize(stop),
                           "Failed to synchronize stop event for custom SGEMM");

            float v3_time = 0.0f; // ms
            checkCudaError(cudaEventElapsedTime(&v3_time, start, stop),
                           "Failed to calculate elapsed time for custom SGEMM");

            // 拷回自定义 SGEMM 结果
            checkCudaError(cudaMemcpy(host_C_v3, device_c_v3, size, cudaMemcpyDeviceToHost),
                           "Failed to copy result matrix from device to host");

            // --------------------------
            // 数值正确性检查：与 cuBLAS 结果逐元素对比
            // --------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_v3[i]) > TOL)
                {
                    error_count++;
                }
            }

            // --------------------------
            // 计算 GFLOPS
            // 单次 GEMM ≈ 2 * N^3 FLOPs（N^3 次乘 + N^3 次加）
            // --------------------------
            float cublas_gflops = (repeat_time * 2.0f * N * N * N) / (cublas_time * 1e6f);
            float v3_gflops = (repeat_time * 2.0f * N * N * N) / (v3_time * 1e6f);

            // 写入 CSV：Matched=1 表示数值结果在 TOL 内与 cuBLAS 一致
            csv_file << N << "," << cublas_gflops << "," << v3_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // 释放资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_a);
            cudaFree(device_b);
            cudaFree(device_c_v3);

            free(host_A);
            free(host_B);
            free(host_C_cublas);
            free(host_C_v3);
        }
        catch (...)
        {
            std::cerr << "Out of memory or error during size: " << N << std::endl;
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v3.csv'" << std::endl;

    return 0;
}
