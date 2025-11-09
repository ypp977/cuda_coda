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
    BM, BN, BK           : Block级别tile大小
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
template <const int BM, const int BN, const int BK, const int row_stride_a, const int row_stride_b>
__device__ void load_from_gmem(int N, int K, const float* A, const float* B, float* As, float* Bs,
                               int inner_row_a, int inner_col_a, int inner_row_b, int inner_col_b)
{
    // ------------------------------
    // 1. 加载A矩阵tile到共享内存(转置存储)
    // ------------------------------
    for (uint off_set = 0; off_set + row_stride_a <= BM; off_set += row_stride_a)
    {
        // 使用float4向量化加载4个float元素
        const float4 tmp =
            reinterpret_cast<const float4*>(&A[(inner_row_a + off_set) * K + inner_col_a * 4])[0];

        // 转置存储到共享内存，优化后续访问模式
        As[(inner_col_a * 4 + 0) * BM + inner_row_a + off_set] = tmp.x;
        As[(inner_col_a * 4 + 1) * BM + inner_row_a + off_set] = tmp.y;
        As[(inner_col_a * 4 + 2) * BM + inner_row_a + off_set] = tmp.z;
        As[(inner_col_a * 4 + 3) * BM + inner_row_a + off_set] = tmp.w;
    }

    // ------------------------------
    // 2. 加载B矩阵tile到共享内存(按原始布局存储)
    // ------------------------------
    for (uint off_set = 0; off_set + row_stride_b <= BK; off_set += row_stride_b)
    {
        // 使用float4向量化加载B矩阵数据
        reinterpret_cast<float4*>(&Bs[(inner_row_b + off_set) * BN + inner_col_b * 4])[0] =
            reinterpret_cast<const float4*>(&B[(inner_row_b + off_set) * N + inner_col_b * 4])[0];
    }
}

/*
------------------------------------------------------------
process_from_smem: 在共享内存中处理矩阵乘法计算
------------------------------------------------------------

模板参数:
    BM, BN, BK     : Block级别tile大小
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
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WMITER,
          const int WNITER, const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ void process_from_smem(float* reg_m, float* reg_n, float* thread_results,
                                  const float* As, const float* Bs, const uint warp_row,
                                  const uint warp_col, const uint thread_row_in_warp,
                                  const uint thread_col_in_warp)
{
    // ------------------------------
    // 1. 沿K维度迭代计算点积
    // ------------------------------
    for (uint dot_idx = 0; dot_idx < BK; ++dot_idx)
    {
        // ------------------------------
        // 2. 加载A矩阵数据到寄存器
        // ------------------------------
        for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
        {
            for (uint i = 0; i < TM; ++i)
            {
                reg_m[w_sub_row_idx * TM + i] =
                    As[(dot_idx * BM) + warp_row * WM + w_sub_row_idx * WSUBM +
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
                    Bs[(dot_idx * BN) + warp_col * WN + w_sub_col_idx * WSUBN +
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
mysgemm_warptiling: 使用warp级别tiling的SGEMM kernel (C = alpha * A * B + beta * C)
------------------------------------------------------------

模板参数:
    BM, BN, BK : Block级别tile大小
    WM, WN     : Warp级别tile大小
    WNITER     : Warp内部列迭代次数
    TM, TN     : Thread级别tile大小
    NUM_THREADS: 每个block的线程数

函数参数:
    M, N, K    : 矩阵维度 (A[MxK], B[KxN], C[MxN])
    alpha, beta: 缩放系数
    A, B       : 输入矩阵 (Row-major)
    C          : 输出矩阵

算法核心:
1. 使用三级tiling结构：Block -> Warp -> Thread
2. 利用warp级别tiling优化共享内存访问
3. 使用寄存器缓存提高计算效率
4. 沿K方向分块处理避免共享内存容量限制
5. 最后将寄存器结果按alpha/beta写回全局内存
------------------------------------------------------------
*/
template <const int BM, const int BN, const int BK, const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    mysgemm_warptiling(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // ------------------------------
    const uint c_row = blockIdx.y; // block 对应的行块索引
    const uint c_col = blockIdx.x; // block 对应的列块索引

    // ------------------------------
    // 2. 计算当前warp在block中的位置
    // ------------------------------
    const uint warp_idx = threadIdx.x / WARP_SIZE; // 当前线程所属warp的索引
    const uint warp_col = warp_idx % (BN / WN);    // warp在block中的列位置
    const uint warp_row = warp_idx / (BN / WN);    // warp在block中的行位置

    // ------------------------------
    // 3. 计算warp内部迭代参数
    // ------------------------------
    constexpr uint WMITER = (WM * WN) / (WARP_SIZE * TM * TN * WNITER); // 行方向迭代次数
    constexpr uint WSUBM = WM / WMITER;                                 // warp子块行大小
    constexpr uint WSUBN = WN / WNITER;                                 // warp子块列大小

    // ------------------------------
    // 4. 计算线程在warp内部的位置
    // ------------------------------
    const uint thread_idx_in_warp = threadIdx.x % WARP_SIZE;           // 线程在warp内的索引
    const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN); // 线程在warp内的列位置
    const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN); // 线程在warp内的行位置

    // ------------------------------
    // 5. 分配共享内存用于缓存 A/B tile
    // ------------------------------
    __shared__ float As[BM * BK]; // A矩阵的共享内存缓存
    __shared__ float Bs[BK * BN]; // B矩阵的共享内存缓存

    // ------------------------------
    // 6. 将 A/B/C 指针偏移到当前 block 对应子矩阵
    // ------------------------------
    A += c_row * BM * K;                                                // A 的行偏移
    B += c_col * BN;                                                    // B 的列偏移
    C += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN; // C 的偏移

    // ------------------------------
    // 7. 计算线程加载 A/B 的行列起点
    // ------------------------------
    const uint inner_row_a = threadIdx.x / (BK / 4);      // A加载的行起点
    const uint inner_col_a = threadIdx.x % (BK / 4);      // A加载的列起点(4元素对齐)
    constexpr uint row_stride_a = (NUM_THREADS * 4) / BK; // A加载的行步长

    const uint inner_row_b = threadIdx.x / (BN / 4);      // B加载的行起点
    const uint inner_col_b = threadIdx.x % (BN / 4);      // B加载的列起点(4元素对齐)
    constexpr uint row_stride_b = NUM_THREADS / (BN / 4); // B加载的行步长

    // ------------------------------
    // 8. 分配寄存器用于缓存计算数据
    // ------------------------------
    float thread_results[WMITER * TM * WNITER * TN] = {0.0}; // 存放线程计算结果
    float reg_m[WMITER * TM] = {0.0};                        // 存放A矩阵数据
    float reg_n[WNITER * TN] = {0.0};                        // 存放B矩阵数据

    // ------------------------------
    // 9. 沿K方向分块处理矩阵乘法
    // ------------------------------
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // 从全局内存加载数据到共享内存
        load_from_gmem<BM, BN, BK, row_stride_a, row_stride_b>(
            N, K, A, B, As, Bs, inner_row_a, inner_col_a, inner_row_b, inner_col_b);
        __syncthreads();

        // 在共享内存中处理矩阵乘法计算
        process_from_smem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            reg_m, reg_n, thread_results, As, Bs, warp_row, warp_col, thread_row_in_warp,
            thread_col_in_warp);

        // 移动到下一个K分块
        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // ------------------------------
    // 10. 将累加结果写回全局内存
    // 按 BLAS 规范执行 alpha * A*B + beta * C
    // 使用 float4 向量化写回
    // ------------------------------
    for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
    {
        for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
        {
            // 计算当前子块在C中的位置
            float* C_interim = C + (w_sub_row_idx * WSUBM) * N + w_sub_col_idx * WSUBN;

            // 向量化写回结果
            for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m += 1)
            {
                for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
                {
                    // 读取C中的原始值
                    float4 tmp = reinterpret_cast<float4*>(
                        &C_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                                   thread_col_in_warp * TN + res_idx_n])[0];

                    // 计算索引
                    const int i = (w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) +
                                  w_sub_col_idx * TN + res_idx_n;

                    // 按BLAS规范计算: C = alpha * A * B + beta * C
                    tmp.x = alpha * thread_results[i + 0] + beta * tmp.x;
                    tmp.y = alpha * thread_results[i + 1] + beta * tmp.y;
                    tmp.z = alpha * thread_results[i + 2] + beta * tmp.z;
                    tmp.w = alpha * thread_results[i + 3] + beta * tmp.w;

                    // 写回结果
                    reinterpret_cast<float4*>(&C_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                                                         thread_col_in_warp * TN + res_idx_n])[0] =
                        tmp;
                }
            }
        }
    }
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
                          "BN must be a multiple of 16*TN to avoid quantization effects");
            static_assert(K10_BM % (16 * K10_TM) == 0,
                          "BM must be a multiple of 16*TM to avoid quantization effects");
            static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BM*BK must be a multiple of 4*256 to vectorize loads");
            static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                          "BN*BK must be a multiple of 4*256 to vectorize loads");

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
