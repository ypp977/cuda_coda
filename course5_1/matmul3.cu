#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf：用于比较浮点数误差
#include <fstream> // std::ofstream：用于写 CSV 文件
#include <iostream>
#include <vector>

// 结果对比时的误差容忍度（单元素）
#define TOL 1e-5f

// 计算二维数组中元素的线性下标：row * leading_dimension + col
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 以 float4 形式访问连续 4 个 float（用于向量化读写）
// 调用方需确保起始地址按 sizeof(float4) 对齐，否则是未定义行为
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// ============================================================
// CUDA runtime API 错误检查
// ============================================================
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================
// cuBLAS API 错误检查
// ============================================================
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS Error: " << msg << " - Status code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------------------------
mysgemm_v3: 高性能手写 SGEMM Kernel
------------------------------------------------------------
计算公式:
    C = alpha * A * B + beta * C

矩阵维度与布局（约定均为 row-major）:
    A: M × K，行跨度 = K
    B: K × N，行跨度 = N
    C: M × N，行跨度 = N

模板参数：
    BLOCK_M, BLOCK_N, BLOCK_K :
        每个 thread block 处理的 C 子块在 M / N / K 方向上的尺寸
    THREAD_M, THREAD_N :
        每个线程负责计算的 C 子块尺寸（行 × 列）

函数参数：
    M, N, K       : 矩阵维度 (A[M×K], B[K×N], C[M×N])
    alpha, beta   : 缩放系数（与 BLAS 约定一致）
    A, B          : 输入矩阵（row-major）
    C             : 输出矩阵（row-major）

整体算法设计（Block × Thread 两级 tiling + 向量化访问）：
    1. 将 C 视为由 BLOCK_M × BLOCK_N 子块（block tile）组成的网格。
       每个 thread block 负责其中一个 C_tile。
    2. 在一个 block 内，使用一维 threadIdx.x，将线程映射为
       (BLOCK_M / THREAD_M) × (BLOCK_N / THREAD_N) 个“线程级子 tile”，
       每个线程负责一个 THREAD_M × THREAD_N 输出子块。
    3. 沿 K 方向以 BLOCK_K 为步长分块（K-block）：
       3.1 将 A 和 B 的当前 K-block 对应子矩阵搬运到 shared memory：
            - A 子块（BLOCK_M × BLOCK_K）在 shared memory 中按“转置布局”存为：
                  shared_a[k, m] = A_tile[m, k]
              即逻辑布局为 [BLOCK_K][BLOCK_M]，行跨度 = BLOCK_M。
              这样在 K 作为外层循环时，以 k 为“行”，m 为“列”，访问连续。
            - B 子块（BLOCK_K × BLOCK_N）在 shared memory 中按 [K][N] 布局存放：
                  shared_b[k, n] = B_tile[k, n]
              即逻辑布局为 [BLOCK_K][BLOCK_N]，行跨度 = BLOCK_N。
            - A/B 的加载均使用 float4 向量化方式，减少指令数与访存指令开销。
       3.2 对每个 k_inner ∈ [0, BLOCK_K)：
            - 当前线程从 shared_a 中取出自己负责的 THREAD_M 行上，
              对应 k_inner 的一列数据（reg_a_tile）。
            - 从 shared_b 中取出自己负责的 THREAD_N 列上，
              对应 k_inner 的一行数据（reg_b_tile）。
            - 在寄存器 accum 中完成 rank-1 更新：
                  accum[m][n] += reg_a_tile[m] * reg_b_tile[n]。
    4. 所有 K-block 累加完成后，将 accum 中的 THREAD_M × THREAD_N 结果
       按 alpha / beta 融合写回 C，对 N 方向使用 float4 向量化写回。

使用前提与约束（由调用方保证）：
    - M 是 BLOCK_M 的整数倍，N 是 BLOCK_N 的整数倍，K 是 BLOCK_K 的整数倍；
    - BLOCK_K、THREAD_N、THREAD_M 均为 4 的倍数（因为内部以 4 为步长做 float4 访问）；
    - blockDim.x == (BLOCK_M / THREAD_M) * (BLOCK_N / THREAD_N)；
    - A/B/C 的起始地址及所有涉及 FETCH_FLOAT4 的偏移均满足 16 字节对齐，
      否则 reinterpret_cast<float4*> 的访问是未定义行为；
    - 该实现没有显式边界检查，如需支持任意尺寸需在 load / store 处补充边界判断。
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float* __restrict__ A,
                           float* __restrict__ B, float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 中的 tile 坐标
    //    block_row_idx / block_col_idx：以 BLOCK_M / BLOCK_N 为单位的 tile 索引
    // ------------------------------
    const int block_col_idx = blockIdx.x;
    const int block_row_idx = blockIdx.y;

    // ------------------------------
    // 2. tile 内线程布局
    //
    //    thread_per_row：在 N 方向上有多少个线程级子 tile（一行有多少个线程负责不同列块）
    //    thread_per_col：在 M 方向上有多少个线程级子 tile（有多少行线程簇）
    //    thread_per_block：该 block 线程总数，应与 blockDim.x 一致
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    // ------------------------------
    // 3. 当前线程在 C_tile 内负责的输出子块左上角“局部坐标”
    //
    //    local_row_idx / local_col_idx：
    //      相对于本 C_tile 左上角的偏移（单位：元素）。
    //
    //    映射规则：
    //      - 线程先沿 N（列）方向排满一行，再沿 M（行）方向换行，
    //        即 threadIdx.x 做 row-major 的 2D 映射。
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M;

    // ------------------------------
    // 4. 共享内存：缓存当前 K-block 上的 A / B 子块
    //
    //    shared_a：
    //      逻辑布局为 [BLOCK_K][BLOCK_M]（行主序，leading_dim = BLOCK_M）：
    //          shared_a[k * BLOCK_M + m] ≡ A_tile[m, k]
    //      即将 A 的子块转置后存储，方便后续以 k 为主维度访问连续内存。
    //
    //    shared_b：
    //      逻辑布局为 [BLOCK_K][BLOCK_N]（行主序，leading_dim = BLOCK_N）：
    //          shared_b[k * BLOCK_N + n] ≡ B_tile[k, n]
    // ------------------------------
    __shared__ float shared_a[BLOCK_K * BLOCK_M];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算每个线程需要加载的 float4 向量数量
    //
    //    总元素个数 / 4 = 总 float4 数量，再均匀划分到每个线程。
    //    这里假设：
    //      BLOCK_M * BLOCK_K 和 BLOCK_N * BLOCK_K 都能被 thread_per_block * 4 整除。
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 为加载 A/B 子块设计每个线程的起始位置与行步长
    //
    //    对 A 子块（BLOCK_M × BLOCK_K）：
    //      a_load_row：
    //        标量视角下，该线程负责的“行起点”索引（0..BLOCK_M-1）
    //      a_load_col：
    //        标量视角下，该线程负责的“列起点”索引，实际按 float4 对齐，
    //        因此内部以 4 为单位向右扩展。
    //      a_load_stride：
    //        在 M 方向（行方向）上的步长，用于 for 循环中覆盖 BLOCK_M 行。
    //
    //    对 B 子块（BLOCK_K × BLOCK_N）：
    //      b_load_row：
    //        该线程负责的 K 方向的行起点
    //      b_load_col：
    //        该线程负责的 N 方向的列起点，同样按 float4 对齐
    //      b_load_stride：
    //        在 K 方向上的步长，用于 for 循环中覆盖 BLOCK_K 行。
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
    //    accum：
    //      当前线程负责的 THREAD_M × THREAD_N C 子块累加结果，初始为 0。
    //
    //    reg_a_vec：
    //      用于暂存从全局内存按 float4 读入的 A 数据（中间缓冲），
    //      随后再拆分为标量写入 shared_a 完成“转置存储”。
    //
    //    reg_a_tile：
    //      在单个 k_inner 下，当前线程负责的 THREAD_M 行上的 A 值（标量）。
    //
    //    reg_b_tile：
    //      在单个 k_inner 下，当前线程负责的 THREAD_N 列上的 B 值（标量）。
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.0f};
    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.0f};
    float reg_a_tile[THREAD_M];
    float reg_b_tile[THREAD_N];

    // ------------------------------
    // 8. 将 A / B / C 指针偏移到当前 block 对应的子矩阵起始位置
    //
    //    A：
    //      指向第 block_row_idx 个 tile 起始行：
    //          行偏移 = block_row_idx * BLOCK_M，列从 0 开始。
    //
    //    B：
    //      指向第 block_col_idx 个 tile 起始列：
    //          行从 0 开始，列偏移 = block_col_idx * BLOCK_N。
    //
    //    C：
    //      指向 C 中对应 tile 左上角：
    //          行偏移 = block_row_idx * BLOCK_M，
    //          列偏移 = block_col_idx * BLOCK_N。
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
        // ------------------------------------------------
        // 9.1 从全局内存加载 A 的当前 K-block 子块到 shared_a
        //
        //   子块的几何范围（以 A 原始视角）：
        //     行：block_row_idx * BLOCK_M .. + BLOCK_M - 1
        //     列：k_block_start .. + BLOCK_K - 1
        //
        //   由于 A 指针在每轮循环结尾会 += BLOCK_K，
        //   这里索引 A[(a_load_row + i), a_load_col..a_load_col+3]
        //   始终落在当前 K-block 对应的列范围。
        //
        //   写入 shared_a 时完成“转置”：
        //     原本是 [BLOCK_M][BLOCK_K]，
        //     在 shared_a 中变为 [BLOCK_K][BLOCK_M]。
        // ------------------------------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            // reg_idx：当前 for-iteration 在 reg_a_vec 中的起始索引（标量）
            const int reg_idx = (i / a_load_stride) * 4;

            // 按 float4 从全局 A 读入四个连续元素：
            //   位置：行 = a_load_row + i，列 = a_load_col..a_load_col+3
            FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

            // 将读到的四个值按“列索引不变、行索引为 (i + a_load_row)”的方式
            // 写入 shared_a，实现 A 子块转置：
            //   shared_a[a_load_col + k, i + a_load_row] = reg_a_vec[reg_idx + k]
            shared_a[OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
            shared_a[OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
            shared_a[OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
            shared_a[OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
        }

        // ------------------------------------------------
        // 9.2 从全局内存加载 B 的当前 K-block 子块到 shared_b
        //
        //   子块几何范围（以 B 原始视角）：
        //     行：k_block_start .. + BLOCK_K - 1
        //     列：block_col_idx * BLOCK_N .. + BLOCK_N - 1
        //
        //   B 指针每轮循环结尾会 += BLOCK_K * N，
        //   这里 B[(b_load_row + i), b_load_col..b_load_col+3] 始终位于当前K-block。
        //   直接按 [BLOCK_K][BLOCK_N] 布局写入 shared_b。
        // ------------------------------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            FETCH_FLOAT4(shared_b[OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
        }

        // 等待所有线程完成 A/B 子块加载
        __syncthreads();

        // ------------------------------------------------
        // 9.3 将 A/B 全局指针推进到下一 K-block 起点
        //
        //   本轮计算仅访问 shared_a / shared_b，
        //   因此这里提前移动 A/B 指针不会影响本轮。
        // ------------------------------------------------
        A += BLOCK_K;     // A：列偏移 BLOCK_K
        B += BLOCK_K * N; // B：行偏移 BLOCK_K（每行 N 元素）

        // ------------------------------------------------
        // 9.4 使用共享内存中的子块进行乘加累积
        //
        //   内层循环 k_inner 遍历 BLOCK_K：
        //     对于每个 k_inner：
        //       1) 从 shared_a 取出当前 k_inner 对应的 A 列上，当前线程负责的 THREAD_M 行；
        //       2) 从 shared_b 取出当前 k_inner 对应的 B 行上，当前线程负责的 THREAD_N 列；
        //       3) 做一次 rank-1 更新 accum += a(k_inner) × b(k_inner)^T。
        // ------------------------------------------------
#pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            // 从 shared_a 中取出当前 k_inner 对应列上、THREAD_M 个 A 元素：
            //   访问：shared_a[k_inner, local_row_idx + m]
            //   按 4 为一组使用 float4 加载到 reg_a_tile
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[m]) =
                    FETCH_FLOAT4(shared_a[OFFSET(k_inner, local_row_idx + m, BLOCK_M)]);
            }

            // 从 shared_b 中取出当前 k_inner 对应行上、THREAD_N 个 B 元素：
            //   访问：shared_b[k_inner, local_col_idx + n]
            //   同样按 4 为一组 float4 加载
#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[n]) =
                    FETCH_FLOAT4(shared_b[OFFSET(k_inner, local_col_idx + n, BLOCK_N)]);
            }

            // 对当前 k_inner 做 rank-1 更新：
            //   对于本线程负责的每个 (m, n)：
            //       accum[m][n] += reg_a_tile[m] * reg_b_tile[n]
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

        // 所有线程完成本轮 K-block 的乘加后，再进入下一轮
        __syncthreads();
    }

    // ------------------------------
    // 10. 将累加结果写回 C（N 方向采用 float4 向量化写回）
    //
    //     本 tile 在 C 中的左上角由 C 的基址给出，
    //     当前线程负责的相对坐标为 (local_row_idx + m, local_col_idx + n)。
    //     线性下标：
    //       row = local_row_idx + m
    //       col = local_col_idx + n
    //       idx = row * N + col
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

// 向上取整除法：常用于根据 tile 尺寸计算 grid 维度
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// 生成测试矩阵大小（如需多尺寸测试，可在此返回多个 N）
std::vector<int> generateSizes()
{
    return {4096};
}

// ============================================================
// 主函数：对比 cuBLAS 与 mysgemm_v3 的性能和正确性
// ============================================================
int main()
{
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "Failed to set CUDA device");

    std::vector<int> sizes = generateSizes();

    // CSV 字段说明：
    //   Size           : 矩阵边长 N（N×N）
    //   CUBLAS_GFLOPS  : cuBLAS 实测 GFLOPS
    //   MySGEMM_FLOPS  : mysgemm_v3 实测 GFLOPS
    //   Matched        : 1 表示与 cuBLAS 在 TOL 内一致，0 表示存在超过 TOL 的差异
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // 当前只测试单个尺寸 N=4096，如需多尺寸可在 generateSizes 中扩展
    for (int N : sizes)
    {
        std::cout << "Testing Size: " << N << std::endl;

        // A/B/C 均为 N×N，row-major
        size_t size = static_cast<size_t>(N) * N * sizeof(float);

        // --------------------------
        // Host 端内存分配
        // --------------------------
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_v3 = (float*)malloc(size);

        // --------------------------
        // Device 端内存分配
        // --------------------------
        float *device_a, *device_b, *device_c_v3;
        checkCudaError(cudaMalloc(&device_a, size),
                       "Failed to allocate device memory for matrix A");
        checkCudaError(cudaMalloc(&device_b, size),
                       "Failed to allocate device memory for matrix B");
        checkCudaError(cudaMalloc(&device_c_v3, size),
                       "Failed to allocate device memory for matrix C");

        try
        {
            // --------------------------
            // 初始化 A、B：
            //   A 全 1.0f，B 全 2.0f
            //   从数学上看，C_ij = Σ_k (1 * 2) = 2 * N。
            //   这在后面既可作为正确性参考，也弱化了“列主 / 行主”差异的影响。
            // --------------------------
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // 拷贝 A / B 到 GPU
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
            // cuBLAS 预热
            //
            // 说明：
            //   - cuBLAS 从语义上假定矩阵是 column-major；
            //   - 这里的 device_a/device_b 按 row-major 存储；
            //   - 但由于 A、B 填充为常数矩阵，乘积 C 中所有元素相等（2*N），
            //     因此即使在“解释方式”上有差异，数值结果仍然是一致的。
            //   - 基于这一特性，可以将 cuBLAS 结果安全地当作参考值，
            //     用于验证手写 kernel 的正确性和做性能对比。
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

            // 为自定义 SGEMM 做准备：C 清零
            checkCudaError(cudaMemset(device_c_v3, 0, size),
                           "Failed to initialize matrix C with zeros");

            // --------------------------
            // 自定义 SGEMM 的执行配置
            //
            //   使用配置：
            //     BLOCK_M = BLOCK_N = 128，BLOCK_K = 8，
            //     THREAD_M = THREAD_N = 8。
            //
            //   含义：
            //     - 每个 block 负责 128×128 的 C_tile；
            //     - blockDim.x = 256：
            //         thread_per_row = 128 / 8 = 16
            //         thread_per_col = 128 / 8 = 16
            //         16 * 16 = 256；
            //     - 每个线程输出 8×8 个元素。
            //
            //   对 N = 4096：
            //     4096 / 128 = 32 → gridDim = (32, 32)，完整覆盖 C。
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
            // 数值正确性检查：
            //   与 cuBLAS 结果逐元素比较，允许 TOL 误差。
            //   为避免输出过多，仅统计最多 10 个误差点。
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
            // 计算 GFLOPS：
            //   单次 GEMM ≈ 2 * N^3 FLOPs（N^3 次乘法 + N^3 次加法）
            //   总 FLOPs = repeat_time * 2 * N^3
            //   时间单位为毫秒，所以除以 (time_ms * 1e6) 得到 GFLOPS。
            // --------------------------
            float cublas_gflops = (repeat_time * 2.0f * N * N * N) / (cublas_time * 1e6f);
            float v3_gflops = (repeat_time * 2.0f * N * N * N) / (v3_time * 1e6f);

            // 写入 CSV，Matched=1 表示数值结果在 TOL 内与 cuBLAS 一致
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
            // 此处兜底异常（如极端 OOM 或其他错误），记录占位信息方便后续定位
            std::cerr << "Out of memory or error during size: " << N << std::endl;
            csv_file << N << ",OOM,OOM,0" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v3.csv'" << std::endl;

    return 0;
}
