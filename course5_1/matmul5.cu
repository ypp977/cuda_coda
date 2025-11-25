#include <cmath>          // fabsf() 用于结果误差比较
#include <cublas_v2.h>    // cuBLAS 库头文件，用于调用高性能矩阵乘法
#include <cuda_runtime.h> // CUDA 运行时 API
#include <fstream>        // 用于将结果写入 CSV
#include <iostream>       // 输入输出
#include <vector>

#define BLOCK_SIZE 128 // 未使用但保留的宏：线程块大小（可用于其他版本）
#define TOL 1e-5f      // 误差容忍度：用于判断 CPU / GPU 结果是否一致

// CUDA 错误检查工具函数：如果 CUDA API 调用失败，打印错误并退出
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// cuBLAS 错误检查工具函数：检查 cuBLAS API 调用是否成功
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
load_from_gmem: 从全局内存加载数据到共享内存
------------------------------------------------------------

模板参数:
    BLOCK_M, BLOCK_N, BLOCK_K           : Block级别tile大小
    row_stride_a, row_stride_b : 加载A和B时的行步长

函数参数:
    N, K         : 矩阵维度
    A, B         : 输入矩阵 (全局内存)
    As, Bs       : 共享内存中的矩阵缓存
    inner_row_a, inner_col_a : 当前线程在A tile中的加载位置
    inner_row_b, inner_col_b : 当前线程在B tile中的加载位置

算法核心:
1. 使用float4向量化加载提高内存带宽利用率
2. 每个线程负责加载一部分数据到共享内存
3. As采用转置存储以优化后续访问模式
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int row_stride_a,
          const int row_stride_b>
__device__ void load_from_gmem(int N, int K, const float* A, const float* B, float* As, float* Bs,
                               int inner_row_a, int inner_col_a, int inner_row_b, int inner_col_b)
{
    // ------------------------------
    // 1. 加载A矩阵tile到共享内存(转置存储)
    // ------------------------------
    for (uint off_set = 0; off_set + row_stride_a <= BLOCK_M; off_set += row_stride_a)
    {
        // 使用float4向量化加载4个float元素
        const float4 tmp =
            reinterpret_cast<const float4*>(&A[(inner_row_a + off_set) * K + inner_col_a * 4])[0];

        // 转置存储到共享内存，优化后续访问模式
        As[(inner_col_a * 4 + 0) * BLOCK_M + inner_row_a + off_set] = tmp.x;
        As[(inner_col_a * 4 + 1) * BLOCK_M + inner_row_a + off_set] = tmp.y;
        As[(inner_col_a * 4 + 2) * BLOCK_M + inner_row_a + off_set] = tmp.z;
        As[(inner_col_a * 4 + 3) * BLOCK_M + inner_row_a + off_set] = tmp.w;
    }

    // ------------------------------
    // 2. 加载B矩阵tile到共享内存(按原始布局存储)
    // ------------------------------
    for (uint off_set = 0; off_set + row_stride_b <= BLOCK_K; off_set += row_stride_b)
    {
        // 使用float4向量化加载B矩阵数据
        reinterpret_cast<float4*>(&Bs[(inner_row_b + off_set) * BLOCK_N + inner_col_b * 4])[0] =
            reinterpret_cast<const float4*>(&B[(inner_row_b + off_set) * N + inner_col_b * 4])[0];
    }
}

/*
------------------------------------------------------------
process_from_smem: 在共享内存中处理矩阵乘法计算
------------------------------------------------------------

模板参数:
    BLOCK_M, BLOCK_N, BLOCK_K     : Block级别tile大小
    WM, WN         : Warp级别tile大小
    WMITER, WNITER : Warp内部迭代次数
    WSUBM, WSUBN   : Warp子tile大小
    TM, TN         : Thread级别tile大小

函数参数:
    reg_m, reg_n         : 存放线程从共享内存加载的数据的寄存器
    thread_results       : 存放线程计算结果的寄存器
    As, Bs               : 共享内存中的矩阵数据
    warp_row, warp_col   : 当前warp在block中的行列位置
    thread_row_in_warp, thread_col_in_warp : 当前线程在warp中的行列位置

算法核心:
1. 沿K维度进行点积计算
2. 每个线程从共享内存加载数据到寄存器
3. 在寄存器中进行计算以提高性能
4. 累加结果到thread_results
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN, const int TM,
          const int TN>
__device__ void process_from_smem(float* reg_m, float* reg_n, float* thread_results,
                                  const float* As, const float* Bs, const uint warp_row,
                                  const uint warp_col, const uint thread_row_in_warp,
                                  const uint thread_col_in_warp)
{
    // ------------------------------
    // 1. 沿K维度迭代计算点积
    // ------------------------------
    for (uint dot_idx = 0; dot_idx < BLOCK_K; ++dot_idx)
    {
        // ------------------------------
        // 2. 加载A矩阵数据到寄存器
        // ------------------------------
        for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                reg_m[w_sub_row_idx * TM + i] =
                    As[(dot_idx * BLOCK_M) + warp_row * WM + w_sub_row_idx * WSUBM +
                       thread_row_in_warp * TM + i];
            }
        }

        // ------------------------------
        // 3. 加载B矩阵数据到寄存器
        // ------------------------------
        for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
        {
            for (uint i = 0; i < TN; ++i)
            {
                reg_n[w_sub_col_idx * TN + i] =
                    Bs[(dot_idx * BLOCK_N) + warp_col * WN + w_sub_col_idx * WSUBN +
                       thread_col_in_warp * TN + i];
            }
        }

        // ------------------------------
        // 4. 在寄存器中计算矩阵乘法累加
        // ------------------------------
        for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
        {
            for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
            {
                for (uint res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
                {
                    for (uint res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                    {
                        thread_results[(w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                       (w_sub_col_idx * TN) + res_idx_n] +=
                            reg_m[w_sub_row_idx * TM + res_idx_m] *
                            reg_n[w_sub_col_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}

// Warp大小常量
constexpr int WARP_SIZE = 32;

/*
------------------------------------------------------------
模块名：mysgemm_warptiling
------------------------------------------------------------
1. 功能：
   - 使用三层 tiling（Block / Warp / Thread）实现高性能 SGEMM：
       C = alpha * A * B + beta * C
   - 矩阵均为 row-major（行主序）存储。

2. 模板参数：
   - BLOCK_M, BLOCK_N, BLOCK_K：
     Block 级 tile 尺寸。单个 block 负责计算
     C 的一个 BLOCK_M（行）× BLOCK_N（列）子矩阵，
     并沿 K 方向以 BLOCK_K 为深度分块累加。
   - WARP_M, WARP_N：
     Warp 级 tile 尺寸。单个 warp 在 block 内负责计算
     C 的一个 WARP_M × WARP_N 子矩阵。
   - WARP_N_ITER：
     Warp 在 N 方向对子 tile 的迭代次数，
     用于覆盖完整的 WARP_N 宽度（一次迭代处理 WARP_SUB_TILE_N 列）。
   - THREAD_TILE_M, THREAD_TILE_N：
     Thread 级 tile 尺寸。单线程在一次子 tile 迭代中
     负责输出 THREAD_TILE_M × THREAD_TILE_N 个 C 元素。
   - NUM_THREADS：
     每个 block 的线程总数，需为 WARP_SIZE 的整数倍，
     并与 BLOCK/WARP/THREAD tile 的拆分关系匹配。

3. 函数参数：
   - M, N, K：
     矩阵维度，A[M × K]，B[K × N]，C[M × N]，均为 row-major。
   - alpha, beta：
     标量系数，最终计算：
       C = alpha * (A * B) + beta * C
   - A, B：
     输入矩阵首地址（row-major）：
       · A 行跨度为 K
       · B 行跨度为 N
   - C：
     输出/输入矩阵首地址（row-major），行跨度为 N。

4. 算法步骤（沿 K 方向分块累加）：
   1) 将 C 按 BLOCK_M × BLOCK_N 划分为 block tile，每个 block 负责一个 tile。
   2) block 内将该 tile 进一步划分为多个 WARP_M × WARP_N 的 warp tile。
   3) 对每个 K 分块：
        a. 所有线程协同将 A 的 BLOCK_M × BLOCK_K 子块、
           B 的 BLOCK_K × BLOCK_N 子块搬运到共享内存。
        b. 每个线程从共享内存读取自身负责的 A/B 局部片段到寄存器。
        c. 在寄存器中完成 FMA 累加，更新本线程的 accum_frag。
   4) 所有 K 分块累加完毕后，将 accum_frag 按 alpha/beta 写回 C。

5. 使用注意：
   - BLOCK/WARP/THREAD tile 之间需满足整除与覆盖关系，
     以保证 lane / thread 的映射合法且共享内存访问不越界。
   - 若使用 float4 向量化加载，A/B 对应地址需保证 16B 对齐，
     否则会影响性能或触发未对齐访问。
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
    const uint block_tile_row = blockIdx.x; // block 在 M 方向的 tile 索引
    const uint block_tile_col = blockIdx.y; // block 在 N 方向的 tile 索引

    // warp_id_in_block：
    //   当前线程所在 warp 在 block 内的一维编号（WARP_SIZE=32）
    const uint warp_id_in_block = threadIdx.x / WARP_SIZE;

    // warps_per_block_n：
    //   当前 block 在 N 方向可容纳的 warp tile 数量
    //   每个 warp 计算一个 WARP_M × WARP_N 的 C 子块
    const uint warps_per_block_n = BLOCK_N / WARP_N;

    // warp_tile_row / warp_tile_col：
    //   warp 在 block 内的二维 tile 坐标
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
    //   缓存 A 的一个 BLOCK_M × BLOCK_K 子块（row-major 展平）
    __shared__ float shared_a[BLOCK_M * BLOCK_K];

    // shared_b：
    //   缓存 B 的一个 BLOCK_K × BLOCK_N 子块（row-major 展平）
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 4. 全局内存指针偏移到当前 Block / Warp 对应的子矩阵
    // ------------------------------
    // A 偏移到当前 block 负责的 A tile 左上角：
    //   起始行 = block_tile_row * BLOCK_M，起始列 = 0
    A += block_tile_row * BLOCK_M * K;

    // B 偏移到当前 block 负责的 B tile 左上角：
    //   起始行 = 0，起始列 = block_tile_col * BLOCK_N
    B += block_tile_col * BLOCK_N;

    // C 偏移到当前 warp 负责写回的 C warp tile 左上角：
    //   起始行 = block_tile_row * BLOCK_M + warp_tile_row * WARP_M
    //   起始列 = block_tile_col * BLOCK_N + warp_tile_col * WARP_N
    C += (block_tile_row * BLOCK_M + warp_tile_row * WARP_M) * N + block_tile_col * BLOCK_N +
         warp_tile_col * WARP_N;

    // ------------------------------
    // 5. A 子块加载索引（按 float4 向量化加载布局）
    // ------------------------------
    // 将 BLOCK_M × BLOCK_K 的 A tile 视为：
    //   BLOCK_M 行 × (BLOCK_K / 4) 列 float4 向量
    const uint load_a_row = threadIdx.x / (BLOCK_K / 4); // 负责的“向量行”索引
    const uint load_a_col = threadIdx.x % (BLOCK_K / 4); // 负责的“向量列”索引（float4）

    // load_a_row_stride：
    //   覆盖完整 A tile 时，相邻两轮加载在行方向上的步长（单位：行）
    //   一轮所有线程共加载 NUM_THREADS * 4 个 float，
    //   按 K 方向长度 BLOCK_K 展开，相当于加载 (NUM_THREADS*4 / BLOCK_K) 行。
    constexpr uint load_a_row_stride = (NUM_THREADS * 4) / BLOCK_K;

    // ------------------------------
    // 6. B 子块加载索引（按 float4 向量化加载布局）
    // ------------------------------
    // 将 BLOCK_K × BLOCK_N 的 B tile 视为：
    //   BLOCK_K 行 × (BLOCK_N / 4) 列 float4 向量
    const uint load_b_row = threadIdx.x / (BLOCK_N / 4); // 负责的“向量行”索引
    const uint load_b_col = threadIdx.x % (BLOCK_N / 4); // 负责的“向量列”索引（float4）

    // load_b_row_stride：
    //   覆盖完整 B tile 时，相邻两轮加载在行方向上的步长（单位：行）
    //   一轮中每线程加载一个 float4，
    //   因此一轮可覆盖 NUM_THREADS / (BLOCK_N/4) 行。
    constexpr uint load_b_row_stride = NUM_THREADS / (BLOCK_N / 4);

    // ------------------------------
    // 7. 寄存器片段：C 累加结果 + A / B 局部 tile
    // ------------------------------
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

    // 后续：K 方向循环 + shared_a/shared_b 填充 + warp/thread 级乘加累积 + 写回 C
    // 在完整 kernel 中继续展开。
}

// 生成测试矩阵大小
std::vector<int> generateSizes()
{
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256)
    {
        sizes.push_back(i);
    }
    return sizes;
}

// 整除向上宏
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

int main()
{
    // 生成一组要测试的矩阵尺寸，例如 [256, 512, 768, ..., 8192]
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件，用于记录不同矩阵规模下的性能
    std::ofstream csv_file("sgemm_benchmark_v7.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    // 遍历每一个矩阵规模进行测试
    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        // 计算矩阵占用的总字节数（三个 N×N float 矩阵）
        size_t size = N * N * sizeof(float);

        // 在主机端分配内存
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size); // 用于存放 cuBLAS 计算结果
        float* C_v1 = (float*)malloc(size);     // 用于存放自实现 SGEMM 结果

        // 在 GPU 上分配内存
        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;

        try
        {
            // 初始化主机矩阵，这里使用简单常数，方便验证正确性
            for (int i = 0; i < N * N; ++i)
            {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            // 把 A、B 复制到 GPU
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy B to device failed");

            // 创建 cuBLAS 句柄
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            // 创建用于测量 GPU 时间的 event
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // -------- Warm-up cuBLAS，避免首次调用偏慢 -------- //
            int warpup_time = 10; // 热身次数
            for (int i = 0; i < warpup_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // -------- 正式计时 cuBLAS -------- //
            int repeat_time = 5;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start cublas) failed");
            for (int i = 0; i < repeat_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }

            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop cublas) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize cublas failed");

            // 计算 cuBLAS 花费时间（毫秒）
            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 取回 cuBLAS 结果，用于正确性比较
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_cublas failed");

            // 清空 C，准备测自实现版本
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            // 配置自定义 kernel 的执行参数
            const uint K10_NUM_THREADS = 128;
            const uint K10_BN = 128;
            const uint K10_BM = 128;
            const uint K10_BK = 16;
            const uint K10_WN = 64;
            const uint K10_WM = 64;
            const uint K10_WNITER = 4;
            const uint K10_TN = 4;
            const uint K10_TM = 8;
            dim3 blockDim(K10_NUM_THREADS);

            constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

            // 静态断言检查参数配置的正确性
            static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
            static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);
            static_assert((K10_WM * K10_WN) % (WARP_SIZE * K10_TM * K10_TN * K10_WNITER) == 0);
            constexpr uint K10_WMITER = (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
            static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

            static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                          "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                          "issues during GMEM->SMEM tiling (loading only parts of the "
                          "final row of Bs during each iteraion)");
            static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                          "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                          "issues during GMEM->SMEM tiling (loading only parts of the "
                          "final row of As during each iteration)");
            static_assert(K10_BN % (16 * K10_TN) == 0,
                          "BLOCK_N must be a multiple of 16*TN to avoid quantization effects");
            static_assert(K10_BM % (16 * K10_TM) == 0,
                          "BLOCK_M must be a multiple of 16*TM to avoid quantization effects");
            static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BLOCK_M*BLOCK_K must be a multiple of 4*256 to vectorize loads");
            static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BLOCK_N*BLOCK_K must be a multiple of 4*256 to vectorize loads");

            dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(N, K10_BM));

            // -------- Warm-up 自定义 SGEMM -------- //
            for (int i = 0; i < warpup_time; ++i)
            {
                mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                                   K10_TN, K10_NUM_THREADS>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            cudaDeviceSynchronize();

            // -------- 正式计时自实现 SGEMM -------- //
            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");
            for (int i = 0; i < repeat_time; ++i)
            {
                mysgemm_warptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                                   K10_TN, K10_NUM_THREADS>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");
            checkCudaError(cudaGetLastError(), "cuda get last error failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 取回自实现的 C 进行正确性验证
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");

            // 简单比较前 10 个元素，验证是否数值一致
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i)
            {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
                {
                    error_count++;
                }
            }

            // 计算 GFLOPS（公式：2*N³ / 时间）
            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);         // GFlops

            float ratio = v1_gflops / cublas_gflops;

            // 将结果写入 CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

            // 回收资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C_v1);

            free(A);
            free(B);
            free(C_cublas);
            free(C_v1);
            cudaDeviceSynchronize();
        }
        catch (...)
        {
            // 如果发生内存不足（OOM）则记录并继续下一个尺寸
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

    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'" << std::endl;
    return 0;
}
