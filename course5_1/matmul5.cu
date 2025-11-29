#include <cmath>          // fabsf()：用于结果误差比较
#include <cublas_v2.h>    // cuBLAS 库头文件：高性能矩阵运算（列主序接口）
#include <cuda_runtime.h> // CUDA 运行时 API
#include <fstream>        // 将结果写入 CSV
#include <iostream>       // 标准输入输出
#include <vector>

#define TOL 1e-5f // 误差容忍度：用于判断 cuBLAS 和自实现 SGEMM 结果是否“一致”（绝对误差比较）

// ------------------------------------------------------------
// CUDA 错误检查工具函数：
// 1) 包装 CUDA API 返回值检查
// 2) 如果调用失败，打印错误信息并直接退出进程
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
// cuBLAS 错误检查工具函数：
// 1) 包装 cuBLAS API 返回值检查
// 2) 如果调用失败，打印错误码并直接退出进程
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
模块名：load_from_gmem
------------------------------------------------------------
1. 功能：
   - 从全局内存加载当前 block 负责的 A/B 子块（tile）到共享内存：
       · A 子块：大小为 BLOCK_M × BLOCK_K，在 shared memory 中按“转置布局”存储
                 （逻辑视为 [BLOCK_K][BLOCK_M]），便于后续按 K 维度连续访问。
       · B 子块：大小为 BLOCK_K × BLOCK_N，在 shared memory 中保持 row-major 布局
                 （逻辑视为 [BLOCK_K][BLOCK_N]）。
   - 使用 float4 向量化加载，提升全局内存带宽利用率（要求 K、N 与对齐约束满足）。

2. 模板参数：
   - BLOCK_M, BLOCK_N, BLOCK_K：
       Block 级 tile 尺寸，对应：
         · A tile 大小：BLOCK_M × BLOCK_K
         · B tile 大小：BLOCK_K × BLOCK_N
   - load_a_row_stride：
       A tile 加载行步长（单位：行），即“所有线程一轮加载共同能覆盖的 A 行数”。
   - load_b_row_stride：
       B tile 加载行步长（单位：行），含义同上。

3. 函数参数：
   - N, K：
       全局矩阵维度（A 为 M×K，B 为 K×N），用于计算线性下标。
   - A, B：
       指向当前 block 需要的 A/B 子矩阵首地址（全局内存）。
       要求：
         · BLOCK_K 和 BLOCK_N 必须是 4 的倍数，以保证 float4 对齐。
         · A, B 传入时已按 block 的行/列偏移过（即已指向该 block 所负责的 tile 左上角）。
   - shared_a, shared_b：
       共享内存缓冲区：
         · shared_a 逻辑布局：[BLOCK_K][BLOCK_M]（K-major，转置存储）
         · shared_b 逻辑布局：[BLOCK_K][BLOCK_N]（与原 row-major 一致）
   - load_a_row, load_a_vec_col：
       当前线程在 A tile 中负责加载的“向量行 / 向量列”索引：
         · load_a_row     ∈ [0, BLOCK_M)
         · load_a_vec_col ∈ [0, BLOCK_K/4)
       每个线程一次加载一个 float4，对应 A 中一行上的 4 个连续元素。
   - load_b_row, load_b_vec_col：
       当前线程在 B tile 中负责加载的“向量行 / 向量列”索引：
         · load_b_row     ∈ [0, BLOCK_K)
         · load_b_vec_col ∈ [0, BLOCK_N/4)

4. 算法步骤：
   1) 对于 A：
        - 按行偏移 row_offset = 0, load_a_row_stride, 2*load_a_row_stride, ... 循环，
          直到覆盖 BLOCK_M 行。
        - 每次循环中，每个线程从 A 中加载一段 float4，并在 shared_a 中以
          “按 K 维度为主、按 M 维度为次”的方式转置存储，以减少后续遍历 K 时的 bank conflict。
   2) 对于 B：
        - 按 row_offset = 0, load_b_row_stride, 2*load_b_row_stride, ... 循环，
          覆盖 BLOCK_K 行。
        - 每次循环中，每个线程从 B 中加载一段 float4，并按原 row-major 布局写入 shared_b，
          方便后续按 N 方向访问。

5. 使用注意：
   - 必须保证：
       · load_a_row_stride > 0 且 BLOCK_M % load_a_row_stride == 0
       · load_b_row_stride > 0 且 BLOCK_K % load_b_row_stride == 0
   - reinterpret_cast<float4*> 访问前提：
       · A/B/共享内存地址按 16 字节对齐（通常由 cudaMalloc + 合理 tile 尺寸保证）。
------------------------------------------------------------
*/

template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int load_a_row_stride,
          const int load_b_row_stride>
__device__ void load_from_gmem(int N, int K, const float* __restrict__ A,
                               const float* __restrict__ B, float* __restrict__ shared_a,
                               float* __restrict__ shared_b, int load_a_row, int load_a_vec_col,
                               int load_b_row, int load_b_vec_col)
{
    // ------------------------------
    // 1. 加载 A 矩阵 tile 到共享内存（转置存储）
    // ------------------------------
    // 逻辑视角：
    //   - 全局 A 子块： [row = 0..BLOCK_M-1][col = 0..BLOCK_K-1]
    //   - 每个线程：负责某一行 load_a_row 上的第 load_a_vec_col 个 float4
    //   - 循环 row_offset：在 M 方向上分片，遍历完整 BLOCK_M 行
    //
    // 共享内存布局 shared_a：
    //   - 视为 [BLOCK_K][BLOCK_M]，行跨度为 BLOCK_M：
    //       shared_a[k * BLOCK_M + m] = A(m, k)
    for (uint row_offset = 0; row_offset + load_a_row_stride <= BLOCK_M;
         row_offset += load_a_row_stride)
    {
        const uint global_row = static_cast<uint>(load_a_row + row_offset); // A 中的行索引
        const uint global_col_vec = static_cast<uint>(load_a_vec_col);      // float4 列索引

        // 从全局内存加载 4 个连续元素：A[global_row][global_col_vec*4 .. +3]
        const float4 a_vec =
            reinterpret_cast<const float4*>(&A[global_row * K + global_col_vec * 4])[0];

        // 将 A(m, k) 以转置方式写入 shared_a：
        //   原：A[global_row][base_k + i]
        //   目标：shared_a[(base_k + i) * BLOCK_M + global_row]
        const uint base_k = global_col_vec * 4;
        const uint base_m = global_row;

        shared_a[(base_k + 0) * BLOCK_M + base_m] = a_vec.x;
        shared_a[(base_k + 1) * BLOCK_M + base_m] = a_vec.y;
        shared_a[(base_k + 2) * BLOCK_M + base_m] = a_vec.z;
        shared_a[(base_k + 3) * BLOCK_M + base_m] = a_vec.w;
    }

    // ------------------------------
    // 2. 加载 B 矩阵 tile 到共享内存（保持 row-major 布局）
    // ------------------------------
    // 逻辑视角：
    //   - 全局 B 子块： [row = 0..BLOCK_K-1][col = 0..BLOCK_N-1]
    //   - 每个线程：负责某一行 load_b_row 上的第 load_b_vec_col 个 float4
    //   - 循环 row_offset：在 K 方向分片，遍历完整 BLOCK_K 行
    //
    // 共享内存布局 shared_b：
    //   - 视为 [BLOCK_K][BLOCK_N]，与 B 的 row-major 布局一致：
    //       shared_b[k * BLOCK_N + n] = B(k, n)
    for (uint row_offset = 0; row_offset + load_b_row_stride <= BLOCK_K;
         row_offset += load_b_row_stride)
    {
        const uint global_row = static_cast<uint>(load_b_row + row_offset); // B 中的行索引
        const uint global_col_vec = static_cast<uint>(load_b_vec_col);      // float4 列索引

        // 从全局 B 加载 float4：B[global_row][global_col_vec*4 .. +3]
        const float4 b_vec =
            reinterpret_cast<const float4*>(&B[global_row * N + global_col_vec * 4])[0];

        // 直接按 row-major 写入 shared_b：
        //   shared_b[global_row][global_col_vec*4 .. +3]
        reinterpret_cast<float4*>(&shared_b[global_row * BLOCK_N + global_col_vec * 4])[0] = b_vec;
    }
}

/*
------------------------------------------------------------
模块名：process_from_smem
------------------------------------------------------------
1. 功能：
   - 在共享内存中的 A / B tile 基础上执行矩阵乘法核心 FMA 计算：
       C_warp_tile += A_block_tile × B_block_tile（沿 K 维度展开）
   - 每个线程：
       · 从 shared_a / shared_b 中加载自己负责的 A/B 局部片段到寄存器 a_frag / b_frag
       · 使用寄存器数据对 accum_frag 中的 C 局部结果进行累加。

2. 模板参数：
   - BLOCK_M, BLOCK_N, BLOCK_K：
       Block 级 tile 尺寸，BLOCK_K 对应当前 K 子块长度。
   - WARP_M, WARP_N：
       Warp 级 tile 尺寸：单个 warp 负责的 C 子矩阵大小。
   - WARP_M_ITER, WARP_N_ITER：
       Warp 在 M / N 方向上子 tile 的迭代次数：
         · WARP_M = WARP_M_ITER * WARP_SUB_TILE_M
         · WARP_N = WARP_N_ITER * WARP_SUB_TILE_N
   - WARP_SUB_TILE_M, WARP_SUB_TILE_N：
       单次迭代时，warp 级子 tile 在 M / N 方向的尺寸。
   - THREAD_TILE_M, THREAD_TILE_N：
       单线程在 M / N 方向负责的输出子块尺寸（thread tile）。

3. 函数参数：
   - a_frag, b_frag：
       寄存器缓存：
         · a_frag 长度：WARP_M_ITER * THREAD_TILE_M
         · b_frag 长度：WARP_N_ITER * THREAD_TILE_N
       分别保存当前线程在所有 M/N 子 tile 上的 A/B 扫描值（当前 k_step）。
   - accum_frag：
       寄存器中累加的 C 结果片段，逻辑布局为：
         [WARP_M_ITER * THREAD_TILE_M] × [WARP_N_ITER * THREAD_TILE_N] 拉平的一维数组。
   - shared_a, shared_b：
       共享内存中的 A / B tile 数据：
         · shared_a：逻辑视为 [BLOCK_K][BLOCK_M]，索引：k * BLOCK_M + m
         · shared_b：逻辑视为 [BLOCK_K][BLOCK_N]，索引：k * BLOCK_N + n
   - warp_tile_row, warp_tile_col：
       当前 warp 在 block 级 C tile 中的 warp 级坐标（单位：WARP_M / WARP_N 元素）。
   - thread_tile_row, thread_tile_col：
       当前线程在 warp 子 tile 内的“thread tile 坐标”，决定其负责的 C 局部区域起点。

4. 算法步骤：
   1) 对 K 维度的每一个 k_step（0..BLOCK_K-1）：
        a. 从 shared_a 中加载当前 k_step 上的若干 A 行片段到 a_frag：
             - M 方向按 warp_tile_row / warp_m_iter_idx / thread_tile_row / tti_m 分别定位。
        b. 从 shared_b 中加载当前 k_step 上的若干 B 列片段到 b_frag：
             - N 方向按 warp_tile_col / warp_n_iter_idx / thread_tile_col / tti_n 分别定位。
        c. 遍历所有 (warp_m_iter_idx, warp_n_iter_idx, tti_m, tti_n) 组合，
           对 accum_frag(m, n) 执行一次标量 FMA：
             accum_frag(m, n) += a_frag(m) * b_frag(n)
   2) 经多个 K 子块迭代后，accum_frag 中即得到当前线程负责的 C 局部最终结果。

5. 使用注意：
   - shared_a / shared_b 的布局必须与 load_from_gmem 保持一致：
       · shared_a[k * BLOCK_M + m] 对应 A_tile(m, k)
       · shared_b[k * BLOCK_N + n] 对应 B_tile(k, n)
   - 索引组合：
       warp_tile_row * WARP_M + warp_m_iter_idx * WARP_SUB_TILE_M
       + thread_tile_row * THREAD_TILE_M + tti_m
     必须保证 < BLOCK_M（同理 N 方向 < BLOCK_N），这依赖于模板参数的整除关系。
------------------------------------------------------------
*/

template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int WARP_M,
          const int WARP_N, const int WARP_M_ITER, const int WARP_N_ITER, const int WARP_SUB_TILE_M,
          const int WARP_SUB_TILE_N, const int THREAD_TILE_M, const int THREAD_TILE_N>
__device__ void
process_from_smem(float* __restrict__ a_frag, float* __restrict__ b_frag,
                  float* __restrict__ accum_frag, const float* __restrict__ shared_a,
                  const float* __restrict__ shared_b, const uint warp_tile_row,
                  const uint warp_tile_col, const uint thread_tile_row, const uint thread_tile_col)
{
    // ------------------------------
    // 1. 沿 K 维度迭代：对每一个 k_step 做一次 rank-1 更新
    // ------------------------------
    for (uint k_step = 0; k_step < BLOCK_K; ++k_step)
    {
        // ------------------------------
        // 2. 从 shared_a 加载当前 k_step 上的 A 片段到 a_frag
        // ------------------------------
        // shared_a 布局为 [BLOCK_K][BLOCK_M]（K-major）：
        //   shared_a[k_step * BLOCK_M + row_idx]
        //
        // row_idx 由以下部分组成：
        //   - warp_tile_row * WARP_M              ：当前 warp 在 block 内的 M 起点
        //   - warp_m_iter_idx * WARP_SUB_TILE_M   ：warp 在 M 方向的迭代偏移
        //   - thread_tile_row * THREAD_TILE_M     ：当前线程在子 tile 内的起点
        //   - tti_m                               ：thread tile 内的局部行索引
        for (uint warp_m_iter_idx = 0; warp_m_iter_idx < WARP_M_ITER; ++warp_m_iter_idx)
        {
            for (uint tti_m = 0; tti_m < THREAD_TILE_M; ++tti_m)
            {
                const uint row_in_block = warp_tile_row * WARP_M +
                                          warp_m_iter_idx * WARP_SUB_TILE_M +
                                          thread_tile_row * THREAD_TILE_M + tti_m;

                a_frag[warp_m_iter_idx * THREAD_TILE_M + tti_m] =
                    shared_a[k_step * BLOCK_M + row_in_block];
            }
        }

        // ------------------------------
        // 3. 从 shared_b 加载当前 k_step 上的 B 片段到 b_frag
        // ------------------------------
        // shared_b 布局为 [BLOCK_K][BLOCK_N]（row-major）：
        //   shared_b[k_step * BLOCK_N + col_idx]
        //
        // col_idx 由以下部分组成：
        //   - warp_tile_col * WARP_N              ：当前 warp 在 block 内的 N 起点
        //   - warp_n_iter_idx * WARP_SUB_TILE_N   ：warp 在 N 方向的迭代偏移
        //   - thread_tile_col * THREAD_TILE_N     ：当前线程在子 tile 内的起点
        //   - tti_n                               ：thread tile 内的局部列索引
        for (uint warp_n_iter_idx = 0; warp_n_iter_idx < WARP_N_ITER; ++warp_n_iter_idx)
        {
            for (uint tti_n = 0; tti_n < THREAD_TILE_N; ++tti_n)
            {
                const uint col_in_block = warp_tile_col * WARP_N +
                                          warp_n_iter_idx * WARP_SUB_TILE_N +
                                          thread_tile_col * THREAD_TILE_N + tti_n;

                b_frag[warp_n_iter_idx * THREAD_TILE_N + tti_n] =
                    shared_b[k_step * BLOCK_N + col_in_block];
            }
        }

        // ------------------------------
        // 4. 在寄存器中执行 FMA：accum_frag += a_frag × b_frag
        // ------------------------------
        // 逻辑上：
        //   对所有 (warp_m_iter_idx, warp_n_iter_idx, tti_m, tti_n)：
        //     C_local(m, n) += A_local(m) * B_local(n)
        //
        // 累加结果在 accum_frag 中的线性索引为：
        //   row_idx = warp_m_iter_idx * THREAD_TILE_M + tti_m
        //   col_idx = warp_n_iter_idx * THREAD_TILE_N + tti_n
        //   idx     = row_idx * (WARP_N_ITER * THREAD_TILE_N) + col_idx
        for (uint warp_m_iter_idx = 0; warp_m_iter_idx < WARP_M_ITER; ++warp_m_iter_idx)
        {
            for (uint warp_n_iter_idx = 0; warp_n_iter_idx < WARP_N_ITER; ++warp_n_iter_idx)
            {
                for (uint tti_m = 0; tti_m < THREAD_TILE_M; ++tti_m)
                {
                    const float a_val = a_frag[warp_m_iter_idx * THREAD_TILE_M + tti_m];

                    for (uint tti_n = 0; tti_n < THREAD_TILE_N; ++tti_n)
                    {
                        const float b_val = b_frag[warp_n_iter_idx * THREAD_TILE_N + tti_n];

                        const uint row_idx = warp_m_iter_idx * THREAD_TILE_M + tti_m;
                        const uint col_idx = warp_n_iter_idx * THREAD_TILE_N + tti_n;

                        const uint accum_index = row_idx * (WARP_N_ITER * THREAD_TILE_N) + col_idx;

                        accum_frag[accum_index] += a_val * b_val;
                    }
                }
            }
        }
    }
}

// Warp 大小常量（所有静态映射均假定 WARP_SIZE=32）
constexpr int WARP_SIZE = 32;

/*
------------------------------------------------------------
模块名：mysgemm_warptiling
------------------------------------------------------------
1. 功能：
   - 使用三层 tiling（Block / Warp / Thread）实现单精度矩阵乘法：
       C = alpha * A * B + beta * C
   - 适用于 row-major 存储的 A[M×K]、B[K×N]、C[M×N]。

2. 模板参数：
   - BLOCK_M, BLOCK_N, BLOCK_K：
       Block 级 tile 尺寸。每个 block 负责计算一个 BLOCK_M×BLOCK_N 的 C 子块，
       并沿 K 方向按 BLOCK_K 分块累加。
   - WARP_M, WARP_N：
       Warp 级 tile 尺寸。每个 warp 负责一个 WARP_M×WARP_N 的 C 子块。
   - WARP_N_ITER：
       Warp 在 N 方向子 tile 的迭代次数。
   - THREAD_TILE_M, THREAD_TILE_N：
       每个线程在 M / N 方向上负责输出的局部子块尺寸（thread tile）。
   - NUM_THREADS：
       每个 block 的线程总数，要求是 WARP_SIZE(32) 的整数倍。

3. 函数参数：
   - M, N, K：
       矩阵维度：A 为 M×K，B 为 K×N，C 为 M×N，均为 row-major。
   - alpha, beta：
       标量系数，最终计算公式为：
         C = alpha * (A * B) + beta * C
   - A, B：
       输入矩阵首地址，row-major：
         · A 行跨度为 K
         · B 行跨度为 N
   - C：
       输出/输入矩阵首地址，row-major，行跨度为 N。

4. 算法步骤（沿 K 分块）：
   1) 将 C 按 BLOCK_M×BLOCK_N 划分为 block 级 tile，每个 block 负责一个 tile。
   2) 在 block 内再按 WARP_M×WARP_N 划分为 warp 级 tile，每个 warp 负责一个子块。
   3) 沿 K 方向以 BLOCK_K 为步长循环：
        a) 所有线程协同，从全局内存以 float4 向量化方式将
           当前 BLOCK_M×BLOCK_K 的 A 子块和 BLOCK_K×BLOCK_N 的 B 子块载入 shared memory。
        b) 每个 warp/线程从 shared memory 中取出属于自己负责的局部 A/B 片段到寄存器，
           在寄存器中对 accum_frag 执行 FMA 累加。
   4) 遍历完所有 K 子块后，将 accum_frag 中的结果按 alpha / beta 组合写回到 C。

5. 关键约束与前提（compile-time 静态断言中检查）：
   - 保证以下整除关系成立：
       · BLOCK_M % WARP_M              == 0
       · BLOCK_N % WARP_N              == 0
       · WARP_M * WARP_N
           是 WARP_SIZE * THREAD_TILE_M * THREAD_TILE_N * WARP_M_ITER * WARP_N_ITER 的整数倍
   - BLOCK_K 和 BLOCK_N 必须是 4 的倍数，以满足 float4 访问和向量化加载分工。
   - C 的基址 + 每次写回偏移需保持 16 字节对齐，否则 reinterpret_cast<float4*> 存在未对齐访问风险。
   - THREAD_TILE_N 至少为 4 且为 4 的整数倍（代码内以 tti_n += 4 迭代）。
------------------------------------------------------------
*/

template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int WARP_M,
          const int WARP_N, const int WARP_N_ITER, const int THREAD_TILE_M, const int THREAD_TILE_N,
          const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    mysgemm_warptiling(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    // ------------------------------
    // 1. Block / Warp 级 tile 在 C 中的定位
    // ------------------------------
    // block_tile_row / block_tile_col：
    //   当前 block 负责的 C 子矩阵 tile 的二维索引（单位：BLOCK_M × BLOCK_N）
    const uint block_tile_row = blockIdx.x; // block 在 M 方向的 tile 索引（按行方向分块）
    const uint block_tile_col = blockIdx.y; // block 在 N 方向的 tile 索引（按列方向分块）

    // warp_id_in_block：
    //   当前线程所在 warp 在 block 内的一维编号（WARP_SIZE = 32）
    const uint warp_id_in_block = threadIdx.x / WARP_SIZE;

    // warps_per_block_n：
    //   当前 block 在 N 方向可容纳的 warp tile 数量
    //   每个 warp 计算一个 WARP_M × WARP_N 的 C 子块
    const uint warps_per_block_n = BLOCK_N / WARP_N;

    // warp_tile_row / warp_tile_col：
    //   warp 在 block 内的二维 tile 坐标（以 warp 级 tile 为单位）
    const uint warp_tile_col = warp_id_in_block % warps_per_block_n; // warp 在 N 方向的序号
    const uint warp_tile_row = warp_id_in_block / warps_per_block_n; // warp 在 M 方向的序号

    // ------------------------------
    // 2. Warp 内 tiling：子 tile 与 thread tile 划分
    // ------------------------------
    // WARP_M_ITER：
    //   warp 在 M 方向的子 tile 迭代次数。
    //   约束关系：
    //     WARP_M * WARP_N
    //       = WARP_SIZE * THREAD_TILE_M * THREAD_TILE_N
    //         * WARP_M_ITER * WARP_N_ITER
    //   即：warp 的输出元素总数 = 32 个线程在所有子 tile 迭代中输出元素数之和。
    constexpr uint WARP_M_ITER =
        (WARP_M * WARP_N) / (WARP_SIZE * THREAD_TILE_M * THREAD_TILE_N * WARP_N_ITER);

    // WARP_SUB_TILE_M / WARP_SUB_TILE_N：
    //   warp 单次子 tile 迭代在 M/N 方向覆盖的输出尺寸
    const uint WARP_SUB_TILE_M = WARP_M / WARP_M_ITER; // 单次迭代覆盖的行数
    const uint WARP_SUB_TILE_N = WARP_N / WARP_N_ITER; // 单次迭代覆盖的列数

    // lane_id：
    //   线程在其所属 warp 内的局部编号 [0, WARP_SIZE)
    const uint lane_id = threadIdx.x % WARP_SIZE;

    // thread_tile_col：
    //   当前线程在 warp 子 tile 内的 thread tile 列索引。
    //   N 方向每行需要 T_N = WARP_SUB_TILE_N / THREAD_TILE_N 个 thread tile，
    //   lane_id 先在 N 方向铺满一行，再换行，因此列号取 lane_id % T_N。
    const uint thread_tile_col = lane_id % (WARP_SUB_TILE_N / THREAD_TILE_N);

    // thread_tile_row：
    //   当前线程在 warp 子 tile 内的 thread tile 行索引。
    //   采用 row-major 映射：每行有 T_N 个 thread tile，
    //   所以行号为 lane_id / T_N。
    const uint thread_tile_row = lane_id / (WARP_SUB_TILE_N / THREAD_TILE_N);

    // ------------------------------
    // 3. Block 级共享内存：缓存 A / B 的 K 方向子块
    // ------------------------------
    // shared_a：
    //   缓存 A 的一个 BLOCK_M × BLOCK_K 子块（线性视为 BLOCK_K×BLOCK_M）
    //   几何意义：当前 block 对应的 C 行范围 × 当前 K 子块的列范围
    __shared__ float shared_a[BLOCK_M * BLOCK_K];

    // shared_b：
    //   缓存 B 的一个 BLOCK_K × BLOCK_N 子块（线性视为 BLOCK_K×BLOCK_N）
    //   几何意义：当前 K 子块的行范围 × 当前 block 对应的 C 列范围
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 4. 全局内存指针偏移到当前 Block / Warp 对应的子矩阵
    // ------------------------------
    // A 偏移到当前 block 负责的 A tile 左上角：
    //   起始行 = block_tile_row * BLOCK_M，起始列 = 0
    //   A 为 row-major，行跨度为 K
    A += block_tile_row * BLOCK_M * K;

    // B 偏移到当前 block 负责的 B tile 左上角：
    //   起始行 = 0，起始列 = block_tile_col * BLOCK_N
    //   B 为 row-major，行跨度为 N
    B += block_tile_col * BLOCK_N;

    // C 偏移到当前 warp 负责写回的 C warp tile 左上角：
    //   起始行 = block_tile_row * BLOCK_M + warp_tile_row * WARP_M
    //   起始列 = block_tile_col * BLOCK_N + warp_tile_col * WARP_N
    //   C 为 row-major，行跨度为 N
    C += (block_tile_row * BLOCK_M + warp_tile_row * WARP_M) * N + block_tile_col * BLOCK_N +
         warp_tile_col * WARP_N;

    // ------------------------------
    // 5. A 子块加载索引（按 float4 向量化加载布局）
    // ------------------------------
    // 将 BLOCK_M × BLOCK_K 的 A tile 视为：
    //   BLOCK_M 行 × (BLOCK_K / 4) 列 float4 向量
    // 要求：BLOCK_K % 4 == 0
    const uint load_a_row = threadIdx.x / (BLOCK_K / 4); // 负责的“向量行”索引
    const uint load_a_col = threadIdx.x % (BLOCK_K / 4); // 负责的“向量列”索引（float4）

    // load_a_row_stride：
    //   覆盖完整 A tile 时，相邻两轮加载在行方向上的步长（单位：行）
    //   一轮所有线程共加载 NUM_THREADS * 4 个 float，
    //   按 K 方向长度 BLOCK_K 展开，相当于加载 (NUM_THREADS * 4 / BLOCK_K) 行。
    constexpr uint load_a_row_stride = (NUM_THREADS * 4) / BLOCK_K;

    // ------------------------------
    // 6. B 子块加载索引（按 float4 向量化加载布局）
    // ------------------------------
    // 将 BLOCK_K × BLOCK_N 的 B tile 视为：
    //   BLOCK_K 行 × (BLOCK_N / 4) 列 float4 向量
    // 要求：BLOCK_N % 4 == 0
    const uint load_b_row = threadIdx.x / (BLOCK_N / 4); // 负责的“向量行”索引
    const uint load_b_col = threadIdx.x % (BLOCK_N / 4); // 负责的“向量列”索引（float4）

    // load_b_row_stride：
    //   覆盖完整 B tile 时，相邻两轮加载在行方向上的步长（单位：行）
    //   一轮中每线程加载一个 float4，
    //   因此一轮可覆盖 NUM_THREADS / (BLOCK_N / 4) 行
    //   即 (NUM_THREADS * 4 / BLOCK_N) 行。
    constexpr uint load_b_row_stride = (NUM_THREADS * 4) / BLOCK_N;

    // ------------------------------
    // 7. 寄存器片段：C 累加结果 + A / B 局部 tile
    // ------------------------------
    // 一个 warp 负责的 C tile 大小：
    //      WARP_M × WARP_N
    //      这个 tile 又被拆成（逻辑上）WARP_M_ITER × WARP_N_ITER 个子 tile
    //      每个子 tile 内，每个线程负责 THREAD_TILE_M × THREAD_TILE_N 个输出。
    //
    // accum_frag：
    //   当前线程负责的输出子块累加结果（初始为 0）。
    //   逻辑维度：
    //     M 方向：WARP_M_ITER × THREAD_TILE_M
    //     N 方向：WARP_N_ITER × THREAD_TILE_N
    float accum_frag[WARP_M_ITER * THREAD_TILE_M * WARP_N_ITER * THREAD_TILE_N] = {0.0f};

    // reg_tile_a：
    //   当前线程在一次 K 子块迭代中从 shared_a 读取的 A 局部片段，
    //   覆盖自身在 M 方向所有子 tile 迭代对应的 THREAD_TILE_M 行。
    float reg_tile_a[WARP_M_ITER * THREAD_TILE_M] = {0.0f};

    // reg_tile_b：
    //   当前线程在一次 K 子块迭代中从 shared_b 读取的 B 局部片段，
    //   覆盖自身在 N 方向所有子 tile 迭代对应的 THREAD_TILE_N 列。
    float reg_tile_b[WARP_N_ITER * THREAD_TILE_N] = {0.0f};

    // ------------------------------
    // 8. 沿 K 方向按 BLOCK_K 分块累加
    // ------------------------------
    // k_iter：当前 K 子块在全局矩阵中的起始下标（0, BLOCK_K, 2*BLOCK_K, ...）
    //         注意：这里用 k_iter 作为“逻辑偏移”，而真正的 A/B 指针偏移通过 A += BLOCK_K
    //         和 B += BLOCK_K * N 体现。
    for (uint k_iter = 0; k_iter < (uint)K; k_iter += BLOCK_K)
    {
        // 8.1 从全局内存加载当前 K 子块对应的 A/B tile 到 shared memory
        load_from_gmem<BLOCK_M, BLOCK_N, BLOCK_K, load_a_row_stride, load_b_row_stride>(
            N, K, A, B, shared_a, shared_b, load_a_row, load_a_col, load_b_row, load_b_col);
        __syncthreads(); // 确保当前 K 子块的 A/B 已全部写入 shared memory

        // 8.2 从 shared memory 读取局部 A/B tile 到寄存器，执行 FMA 累加
        process_from_smem<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_M_ITER, WARP_N_ITER,
                          WARP_SUB_TILE_M, WARP_SUB_TILE_N, THREAD_TILE_M, THREAD_TILE_N>(
            reg_tile_a, reg_tile_b, accum_frag, shared_a, shared_b, warp_tile_row, warp_tile_col,
            thread_tile_row, thread_tile_col);

        // 8.3 全局 A/B 指针推进到下一个 K 子块
        //     A：按列方向（K 维度）偏移 BLOCK_K
        A += BLOCK_K;
        //     B：按行方向偏移 BLOCK_K 行，每行跨度为 N
        B += BLOCK_K * N;

        __syncthreads(); // 确保所有线程都完成当前子块计算后再加载下一个子块
    }

    // ------------------------------
    // 9. 将寄存器中累加结果按 alpha/beta 写回到 C
    // ------------------------------
    // 外层两重循环遍历 warp 在 M/N 方向的子 tile 迭代（WARP_M_ITER × WARP_N_ITER）
    for (uint warp_m_iter = 0; warp_m_iter < WARP_M_ITER; warp_m_iter++)
    {
        for (uint warp_n_iter = 0; warp_n_iter < WARP_N_ITER; warp_n_iter++)
        {
            // c_tile_ptr：
            //   指向当前 warp 子 tile（由 warp_m_iter、warp_n_iter 决定）的 C 子块首地址
            float* c_tile_ptr =
                C + (warp_m_iter * WARP_SUB_TILE_M) * N + warp_n_iter * WARP_SUB_TILE_N;

            // 对当前子 tile 内，每个线程负责 THREAD_TILE_M × THREAD_TILE_N 个元素
            for (uint tti_m = 0; tti_m < THREAD_TILE_M; tti_m++)
            {
                // N 方向以 4 为步长做 float4 向量化写回
                // 要求：THREAD_TILE_N 为 4 的倍数
                for (uint tti_n = 0; tti_n < THREAD_TILE_N; tti_n += 4)
                {
                    // 9.1 从 C 中读取旧值（float4 向量化访问）
                    //     这里假设：
                    //       - C 的基址及内部偏移保证 16B 对齐
                    float4 old_c_val = reinterpret_cast<float4*>(
                        &c_tile_ptr[(thread_tile_row * THREAD_TILE_M + tti_m) * N +
                                    thread_tile_col * THREAD_TILE_N + tti_n])[0];

                    // 9.2 计算当前 4 个输出元素在 accum_frag 中的一维索引基址
                    const int frag_index =
                        (warp_m_iter * THREAD_TILE_M + tti_m) * (WARP_N_ITER * THREAD_TILE_N) +
                        (warp_n_iter * THREAD_TILE_N + tti_n);

                    // 9.3 计算新的输出值：alpha * accum + beta * old_c
                    float4 out;
                    out.x = alpha * accum_frag[frag_index + 0] + beta * old_c_val.x;
                    out.y = alpha * accum_frag[frag_index + 1] + beta * old_c_val.y;
                    out.z = alpha * accum_frag[frag_index + 2] + beta * old_c_val.z;
                    out.w = alpha * accum_frag[frag_index + 3] + beta * old_c_val.w;

                    // 9.4 回写到 C（float4 向量化写回）
                    reinterpret_cast<float4*>(
                        &c_tile_ptr[(thread_tile_row * THREAD_TILE_M + tti_m) * N +
                                    thread_tile_col * THREAD_TILE_N + tti_n])[0] = out;
                }
            }
        }
    }
}

// 生成测试矩阵大小：256, 512, ..., 8192
// 仅生成方阵规模（N×N），便于与 cuBLAS 做直接对比
std::vector<int> generate_test_sizes()
{
    std::vector<int> test_sizes;
    for (int n = 256; n <= 8192; n += 256)
    {
        test_sizes.push_back(n);
    }
    return test_sizes;
}

// 整数向上整除：返回 ceil(M / N)
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

/*
------------------------------------------------------------
模块名：主函数 sgemm 基准测试（cuBLAS vs 自实现 SGEMM）
------------------------------------------------------------
1. 功能：
    - 对一组 N×N 单精度矩阵（N ∈ [256, 8192]，步长 256）进行基准测试：
        · 使用 cuBLAS sgemm 作为参考实现和性能基线
        · 使用自定义 mysgemm_warptiling kernel 进行性能和正确性对比
    - 输出字段（CSV）：
        Size           : 矩阵边长 N（N×N）
        CUBLAS_GFLOPS  : cuBLAS 实测 GFLOPS
        MySGEMM_FLOPS  : 自实现 kernel 实测 GFLOPS（字段名保留 FLOPTS，实际单位为 GFLOPS）
        Matched        : 结果是否匹配（1=匹配，0=前 N² 元素中存在误差> TOL）
        Ratio          : 自实现 GFLOPS / cuBLAS GFLOPS

2. 测试策略：
    - 每个尺寸：
        · 初始化 host_a/host_b：A 全 1.0f，B 全 2.0f（方便快速检查正确性）
        · cuBLAS 部分：
            1) 预热 warmup_iters 次，避免首次调用偏慢
            2) 计时 timed_iters 次，取总时间计算 GFLOPS
        · 自实现 kernel 部分：
            1) 同样 warmup_iters 次预热
            2) 计时 timed_iters 次，计算 GFLOPS
            3) 将结果与 cuBLAS 的 C 做逐元素比较（最多统计 10 个误差点）

3. 性能计时：
    - 使用 cudaEvent_t 进行设备端计时，单位 ms。
    - GFLOPS 计算公式：
        总 FLOPs ≈ 2 * N^3 * timed_iters
        GFLOPS = 总 FLOPs / (time_ms * 1e6)

4. cuBLAS 与 row-major 映射说明：
    - cuBLAS 默认以列主序（column-major）解释矩阵：
        C_cublas = A_cuBLAS * B_cuBLAS
    - 本代码中 host/device 数据按 row-major 存储，为了“逻辑上”实现：
        C_row = A_row * B_row
      采用常见技巧：在 row-major 空间中调用列主序 Sgemm 时，对调 A/B 的位置，
      等价于在数学意义上做转置映射（此处使用对称的 N×N 情况，便于简化映射）。
------------------------------------------------------------
*/
int main()
{
    // 1. 生成测试矩阵尺寸列表
    std::vector<int> test_sizes = generate_test_sizes();

    // 打开 CSV 文件，用于记录不同矩阵规模下的性能数据
    std::ofstream csv_out("sgemm_benchmark_v5.csv");
    csv_out << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    // 2. 遍历每一个矩阵规模进行测试
    for (int matrix_size : test_sizes)
    {
        std::cout << "Testing size: " << matrix_size << std::endl;

        // 三个 N×N float 矩阵的字节数（这里只对单个矩阵的 size，后面分别分配）
        size_t matrix_bytes =
            static_cast<size_t>(matrix_size) * static_cast<size_t>(matrix_size) * sizeof(float);

        // 2.1 主机端内存分配
        float* host_a = static_cast<float*>(malloc(matrix_bytes));
        float* host_b = static_cast<float*>(malloc(matrix_bytes));
        float* host_c_cublas = static_cast<float*>(malloc(matrix_bytes));    // cuBLAS 结果
        float* host_c_kernel_v5 = static_cast<float*>(malloc(matrix_bytes)); // 自实现结果

        // 2.2 设备端内存分配
        float *device_a = nullptr, *device_b = nullptr, *device_c_kernel_v5 = nullptr;
        checkCudaError(cudaMalloc(&device_a, matrix_bytes), "cudaMalloc device_a failed");
        checkCudaError(cudaMalloc(&device_b, matrix_bytes), "cudaMalloc device_b failed");
        checkCudaError(cudaMalloc(&device_c_kernel_v5, matrix_bytes),
                       "cudaMalloc device_c_kernel_v5 failed");

        bool had_error = false;

        try
        {
            // 2.3 初始化主机矩阵（host_a=1, host_b=2），方便验证正确性
            for (int i = 0; i < matrix_size * matrix_size; ++i)
            {
                host_a[i] = 1.0f;
                host_b[i] = 2.0f;
            }

            // 将 A、B 从主机复制到设备
            checkCudaError(cudaMemcpy(device_a, host_a, matrix_bytes, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_a to device failed");
            checkCudaError(cudaMemcpy(device_b, host_b, matrix_bytes, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_b to device failed");

            // 2.4 创建 cuBLAS 句柄
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            // 2.5 创建 GPU 计时用 event
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // 预热 / 正式计次数（对 cuBLAS 和自定义 kernel 共用）
            const int warmup_iters = 10; // 预热次数（不计入性能统计）
            const int timed_iters = 50;  // 正式计时次数

            // ==========================================================
            // 3. cuBLAS SGEMM 性能与结果（基线）
            // ==========================================================

            // 3.1 Warm-up cuBLAS，避免首次调用偏慢
            for (int i = 0; i < warmup_iters; ++i)
            {
                // 注意：cuBLAS 默认列主序，此处通过交换 A/B 的位置来适配 row-major 数据布局
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             matrix_size,                   // m
                                             matrix_size,                   // n
                                             matrix_size,                   // k
                                             &alpha, device_b, matrix_size, // B（列主序视角）
                                             device_a, matrix_size,         // A（列主序视角）
                                             &beta, device_c_kernel_v5, matrix_size), // C
                                 "cublasSgemm warmup failed");
            }
            cudaDeviceSynchronize();

            // 3.2 正式计时 cuBLAS
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < timed_iters; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size,
                                             matrix_size, matrix_size, &alpha, device_b,
                                             matrix_size, device_a, matrix_size, &beta,
                                             device_c_kernel_v5, matrix_size),
                                 "cublasSgemm timed failed");
            }

            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            // 计算 cuBLAS 花费时间（毫秒）
            float cublas_time_ms = 0.0f;
            checkCudaError(cudaEventElapsedTime(&cublas_time_ms, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 将 cuBLAS 结果拷回主机用于后续正确性比较
            checkCudaError(
                cudaMemcpy(host_c_cublas, device_c_kernel_v5, matrix_bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy host_c_cublas failed");

            // 在测自实现 kernel 之前，清空设备端 C 缓冲区
            checkCudaError(cudaMemset(device_c_kernel_v5, 0, matrix_bytes),
                           "cudaMemset device_c_kernel_v5 failed");

            // ==========================================================
            // 4. 配置并测试自定义 mysgemm_warptiling kernel
            // ==========================================================

            // 4.1 kernel 配置参数（BLOCK / WARP / THREAD tile 等）
            const uint KERNEL_NUM_THREADS = 128;
            const uint BLOCK_N = 128;
            const uint BLOCK_M = 128;
            const uint BLOCK_K = 16;

            const uint WARP_N = 64;
            const uint WARP_M = 64;
            const uint WARP_N_ITER = 4;

            const uint THREAD_TILE_N = 4;
            const uint THREAD_TILE_M = 8;

            dim3 blockDim(KERNEL_NUM_THREADS);
            constexpr uint NUM_WARPS = KERNEL_NUM_THREADS / WARP_SIZE;

            // 4.2 静态断言：检查各级 tile 的整除约束与向量化对齐约束
            static_assert((BLOCK_N % WARP_N == 0) && (BLOCK_M % WARP_M == 0),
                          "BLOCK_M/BLOCK_N must be integer multiples of WARP_M/WARP_N");

            static_assert((BLOCK_N / WARP_N) * (BLOCK_M / WARP_M) == NUM_WARPS,
                          "Number of warp tiles per block must equal NUM_WARPS");

            static_assert(
                (WARP_M * WARP_N) % (WARP_SIZE * THREAD_TILE_M * THREAD_TILE_N * WARP_N_ITER) == 0,
                "WARP_M * WARP_N must be divisible by warp output per iteration");

            constexpr uint WARP_M_ITER =
                (WARP_M * WARP_N) / (WARP_SIZE * THREAD_TILE_M * THREAD_TILE_N * WARP_N_ITER);

            static_assert((WARP_M % WARP_M_ITER == 0) && (WARP_N % WARP_N_ITER == 0),
                          "WARP_M/WARP_N must be divisible by WARP_M_ITER/WARP_N_ITER");

            static_assert((KERNEL_NUM_THREADS * 4) % BLOCK_K == 0,
                          "NUM_THREADS*4 must be multiple of BLOCK_K "
                          "to vectorize GMEM->SMEM loads for A");

            static_assert((KERNEL_NUM_THREADS * 4) % BLOCK_N == 0,
                          "NUM_THREADS*4 must be multiple of BLOCK_N "
                          "to vectorize GMEM->SMEM loads for B");

            static_assert(BLOCK_N % (16 * THREAD_TILE_N) == 0,
                          "BLOCK_N must be a multiple of 16*THREAD_TILE_N "
                          "for C write-back tiling");

            static_assert(BLOCK_M % (16 * THREAD_TILE_M) == 0,
                          "BLOCK_M must be a multiple of 16*THREAD_TILE_M "
                          "for C write-back tiling");

            static_assert((BLOCK_M * BLOCK_K) % (4 * KERNEL_NUM_THREADS) == 0,
                          "BLOCK_M*BLOCK_K must be a multiple of 4*NUM_THREADS "
                          "to evenly distribute A loads");

            static_assert((BLOCK_N * BLOCK_K) % (4 * KERNEL_NUM_THREADS) == 0,
                          "BLOCK_N*BLOCK_K must be a multiple of 4*NUM_THREADS "
                          "to evenly distribute B loads");

            // 4.3 计算网格维度：每个 block 负责 BLOCK_M×BLOCK_N 的 C 子矩阵
            // 在本基准测试中仅测试 N×N 方阵，因此 x/y 方向使用同一个 matrix_size
            dim3 gridDim(CEIL_DIV(matrix_size, BLOCK_N),  // blockIdx.x：N 方向 tile 数
                         CEIL_DIV(matrix_size, BLOCK_M)); // blockIdx.y：M 方向 tile 数
            // 注意：kernel 内把 blockIdx.x 视作 M 方向、blockIdx.y 视作 N 方向；
            //       在 N=M 的测试场景下两者数值相同，不影响结果。

            // 4.4 Warm-up 自定义 SGEMM
            for (int i = 0; i < warmup_iters; ++i)
            {
                mysgemm_warptiling<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_N_ITER,
                                   THREAD_TILE_M, THREAD_TILE_N, KERNEL_NUM_THREADS>
                    <<<gridDim, blockDim>>>(matrix_size, matrix_size, matrix_size, alpha, device_a,
                                            device_b, beta, device_c_kernel_v5);
            }
            cudaDeviceSynchronize();

            // 4.5 正式计时自实现 SGEMM
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start kernel) failed");
            for (int i = 0; i < timed_iters; ++i)
            {
                mysgemm_warptiling<BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_N_ITER,
                                   THREAD_TILE_M, THREAD_TILE_N, KERNEL_NUM_THREADS>
                    <<<gridDim, blockDim>>>(matrix_size, matrix_size, matrix_size, alpha, device_a,
                                            device_b, beta, device_c_kernel_v5);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop kernel) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize kernel failed");
            checkCudaError(cudaGetLastError(), "cuda get last error failed");

            float kernel_time_ms = 0.0f;
            checkCudaError(cudaEventElapsedTime(&kernel_time_ms, start, stop),
                           "cudaEventElapsedTime kernel failed");

            // 将自实现 C 结果拷回主机用于正确性验证
            checkCudaError(cudaMemcpy(host_c_kernel_v5, device_c_kernel_v5, matrix_bytes,
                                      cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_c_kernel_v5 failed");

            // ==========================================================
            // 5. 正确性检查与 GFLOPS 计算
            // ==========================================================
            int mismatch_count = 0;
            const int max_mismatches = 10;

            // 逐元素比较 cuBLAS 与 kernel 结果，最多记录 max_mismatches 处误差
            for (int i = 0; i < matrix_size * matrix_size && mismatch_count < max_mismatches; ++i)
            {
                if (fabsf(host_c_cublas[i] - host_c_kernel_v5[i]) > TOL)
                {
                    mismatch_count++;
                }
            }

            // GFLOPS 计算（2*N^3 FLOPs / 时间）
            float cublas_gflops = timed_iters * 2.0f * matrix_size * matrix_size * matrix_size /
                                  (cublas_time_ms * 1e6f);

            float kernel_gflops = timed_iters * 2.0f * matrix_size * matrix_size * matrix_size /
                                  (kernel_time_ms * 1e6f);

            float perf_ratio = kernel_gflops / cublas_gflops;

            // 写入 CSV：Size, cuBLAS, MySGEMM, Matched, Ratio
            csv_out << matrix_size << "," << cublas_gflops << "," << kernel_gflops << ","
                    << (mismatch_count == 0 ? "1" : "0") << "," << perf_ratio << std::endl;

            // 6. 清理当前尺寸的资源（正常路径）
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_a);
            cudaFree(device_b);
            cudaFree(device_c_kernel_v5);

            free(host_a);
            free(host_b);
            free(host_c_cublas);
            free(host_c_kernel_v5);
            cudaDeviceSynchronize();
        }
        catch (...)
        {
            // 如果发生异常（包括可能的 OOM），记录并继续下一个尺寸
            std::cerr << "Error or out-of-memory during testing size: " << matrix_size << std::endl;
            had_error = true;
        }

        if (!had_error)
        {
            std::cout << "Finished size: " << matrix_size << std::endl;
        }
        else
        {
            // 遇到异常时，CSV 中记录为 OOM/ERROR（具体错误未区分）
            csv_out << matrix_size << ",OOM,OOM,0,0" << std::endl;
        }
    }

    csv_out.close();

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v5.csv'" << std::endl;
    return 0;
}
