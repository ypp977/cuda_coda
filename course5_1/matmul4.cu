#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf：用于浮点数误差比较
#include <fstream> // std::ofstream：写 CSV
#include <iostream>
#include <vector>

// 结果校验时的误差容忍度
#define TOL 1e-5f

// 计算二维数组中元素的线性下标：row * leading_dimension + col
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// 以 float4 形式访问连续 4 个 float，用于向量化读写
// 调用方需保证地址按 sizeof(float4) 对齐
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 检查 CUDA API 返回值，如果出错则打印信息并退出
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 返回值，如果出错则打印信息并退出
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS ERROR: " << msg << " - status code " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------------------------
mysgemm_v4: 手写高性能 SGEMM kernel (C = alpha * A * B + beta * C)
------------------------------------------------------------

模板参数:
    BLOCK_M, BLOCK_N, BLOCK_K : Block 级别 tile 尺寸
    THREAD_M, THREAD_N        : 每个线程负责的 C 子块尺寸

函数参数:
    M, N, K : 矩阵维度 (A[M×K], B[K×N], C[M×N])
    alpha, beta : 缩放系数（与 BLAS 规范一致）
    A, B        : 输入矩阵 (假定 Row-major 存储)
    C           : 输出矩阵 (Row-major)

算法核心结构:
1. 将 C 按 BLOCK_M×BLOCK_N 分块，每个 block 负责一个 C tile。
2. block 内线程按 THREAD_M×THREAD_N 划分，每个线程负责一个小的 C 子块。
3. 使用共享内存缓存当前 K-block 上的 A/B 子块，并使用双缓冲（ping-pong）。
   - A 在共享内存中按 [BLOCK_K][BLOCK_M] 转置存储，便于按行连续访问。
   - B 在共享内存中按 [BLOCK_K][BLOCK_N] 存储。
4. 使用 float4 向量化从全局内存加载到寄存器，再写入共享内存。
5. 在寄存器中对线程负责的 C 子块进行累加（accum）。
6. 沿 K 方向以 BLOCK_K 为步长遍历所有子块。
7. 最后将寄存器累加结果按 alpha/beta 融合写回全局内存 C。

调用前提（不在 kernel 内做边界检查）：
- M 是 BLOCK_M 的整数倍，N 是 BLOCK_N 的整数倍，K 是 BLOCK_K 的整数倍；
- BLOCK_K、THREAD_M、THREAD_N 为 4 的倍数（内部以 4 为步长做 float4 访问）；
- blockDim.x == (BLOCK_M / THREAD_M) * (BLOCK_N / THREAD_N)；
- 所有参与 FETCH_FLOAT4 的地址都满足 16 字节对齐。
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void __launch_bounds__(256)
    mysgemm_v4(int M, int N, int K, float alpha, float* __restrict__ A, float* __restrict__ B,
               float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标（以 tile 为单位）
    // ------------------------------
    const int block_col_idx = blockIdx.x; // C 中列方向第几个 BLOCK_N
    const int block_row_idx = blockIdx.y; // C 中行方向第几个 BLOCK_M

    // ------------------------------
    // 2. tile 内线程划分
    //    thread_per_row: 沿 N 方向有多少个“线程级子块”
    //    thread_per_col: 沿 M 方向有多少个“线程级子块”
    //    thread_per_block: block 中线程总数（需与 blockDim.x 一致）
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    // ------------------------------
    // 3. 当前线程在 tile 内负责 C 子块的左上角局部坐标
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M;

    // ------------------------------
    // 4. 共享内存：缓存 A/B tile（双缓冲）
    //    shared_a[2]：两个 A 缓冲区，每个 shape 为 [BLOCK_K][BLOCK_M]（leading_dim = BLOCK_M）
    //    shared_b[2]：两个 B 缓冲区，每个 shape 为 [BLOCK_K][BLOCK_N]（leading_dim = BLOCK_N）
    // ------------------------------
    __shared__ float shared_a[2][BLOCK_K * BLOCK_M];
    __shared__ float shared_b[2][BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 每个线程负责加载的 float4 数量（A/B），用于向量化加载
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 为加载 A/B 子块计算每个线程的起点与步长
    //
    //    对 A：
    //      a_load_row: 以 float 为单位的行起点（在 A 子块内部）
    //      a_load_col: 以 float 为单位的列起点（一次加载 4 个 float，故后面乘 4）
    //      a_load_stride: 遍历 BLOCK_M 行时在 M 方向上的步长
    //
    //    对 B：
    //      b_load_row / b_load_col / b_load_stride 类似含义
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
    //    accum        ：当前线程负责的 THREAD_M×THREAD_N 子块的累加结果
    //    reg_a_vec/b_vec：从全局内存读入（float4）后临时缓存在寄存器中的 A/B 数据
    //    reg_a_tile[2] / reg_b_tile[2]：
    //        在“当前 tile 的 K 内循环”中，为了减少从共享内存的标量读取，
    //        使用 2 个寄存器缓冲区做 ping-pong（bk%2 选择当前/下一组）。
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.0f};

    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.0f};
    float reg_b_vec[4 * vec4_load_per_thread_b] = {0.0f};

    float reg_a_tile[2][THREAD_M];
    float reg_b_tile[2][THREAD_N];

    // ------------------------------
    // 8. 将 A/B/C 指针偏移到当前 block 对应的子矩阵起点（全局坐标）
    //
    //    A 指向：A(block_row_idx * BLOCK_M, 0)
    //    B 指向：B(0, block_col_idx * BLOCK_N)
    //    C 指向：C(block_row_idx * BLOCK_M, block_col_idx * BLOCK_N)
    // ------------------------------
    A = &A[block_row_idx * BLOCK_M * K];
    B = &B[block_col_idx * BLOCK_N];
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N];

    // ------------------------------
    // 9. 预加载第一块 A/B tile 到共享内存（写入 shared_a[0] / shared_b[0]）
    // ------------------------------
#pragma unroll
    for (int i = 0; i < BLOCK_M; i += a_load_stride)
    {
        // reg_idx：本线程在当前循环步内使用的 reg_a_vec 起始下标（以标量计）
        int reg_idx = (i / a_load_stride) * 4;

        // 从全局 A 中以 float4 方式加载：
        //   读取 A[a_load_row + i, a_load_col ... a_load_col+3]
        FETCH_FLOAT4(reg_a_vec[reg_idx]) = FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

        // 写入 shared_a[0] 时进行“转置存储”：
        //   shared_a[0] 视为 [BLOCK_K][BLOCK_M]，leading_dim = BLOCK_M
        shared_a[0][OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
        shared_a[0][OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
        shared_a[0][OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
        shared_a[0][OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_K; i += b_load_stride)
    {
        // B 子块在共享内存中保持 [BLOCK_K][BLOCK_N] 布局
        FETCH_FLOAT4(shared_b[0][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
            FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
    }
    __syncthreads();

    // ------------------------------
    // 10. 从共享内存加载第一轮 k 内循环所需的 A/B 行到寄存器
    //     此处使用 reg_a_tile[0] / reg_b_tile[0] 作为初始 ping 缓冲
    // ------------------------------
#pragma unroll
    for (int m = 0; m < THREAD_M; m += 4)
    {
        FETCH_FLOAT4(reg_a_tile[0][m]) =
            FETCH_FLOAT4(shared_a[0][OFFSET(0, local_row_idx + m, BLOCK_M)]);
    }

#pragma unroll
    for (int n = 0; n < THREAD_N; n += 4)
    {
        FETCH_FLOAT4(reg_b_tile[0][n]) =
            FETCH_FLOAT4(shared_b[0][OFFSET(0, local_col_idx + n, BLOCK_N)]);
    }

    // ------------------------------
    // 11. 沿 K 方向主循环（双缓冲：shared_a[0/1], shared_b[0/1]）
    // ------------------------------
    int write_index = 1; // 当前要写入的共享内存缓冲区编号（0/1）
    int load_index;      // 当前从共享内存读取的缓冲区编号（0/1）
    int k = 0;           // 当前已经处理的 K 元素数

    do
    {
        k += BLOCK_K;

        // ---- 11.1 如果后面还有 K 块，则预先从全局内存加载下一块 A/B 到寄存器 ----
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = (i / a_load_stride) * 4;
                FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_load_row + i, k + a_load_col, K)]);
            }

#pragma unroll
            for (int i = 0; i < BLOCK_K; i += b_load_stride)
            {
                int reg_idx = (i / b_load_stride) * 4;
                FETCH_FLOAT4(reg_b_vec[reg_idx]) =
                    FETCH_FLOAT4(B[OFFSET(k + b_load_row + i, b_load_col, N)]);
            }
        }

        // 当前从哪个共享缓冲读取：与写缓冲相反
        load_index = write_index ^ 1;

        // ---- 11.2 在当前 tile 的 BLOCK_K 个 k_inner 上做计算（除最后一列外） ----
#pragma unroll
        for (int bk = 0; bk < BLOCK_K - 1; bk++)
        {
            // 11.2.1 预取下一 k_inner 的 A 行到寄存器
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[(bk + 1) % 2][m]) =
                    FETCH_FLOAT4(shared_a[load_index][OFFSET(bk + 1, local_row_idx + m, BLOCK_M)]);
            }

            // 11.2.2 预取下一 k_inner 的 B 行到寄存器
#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[(bk + 1) % 2][n]) =
                    FETCH_FLOAT4(shared_b[load_index][OFFSET(bk + 1, local_col_idx + n, BLOCK_N)]);
            }

            // 11.2.3 使用当前 k_inner 的 reg_a_tile[bk%2] / reg_b_tile[bk%2] 进行 rank-1 更新
#pragma unroll
            for (int m = 0; m < THREAD_M; m++)
            {
#pragma unroll
                for (int n = 0; n < THREAD_N; n++)
                {
                    accum[m][n] += reg_a_tile[bk % 2][m] * reg_b_tile[bk % 2][n];
                }
            }
        }

        // ---- 11.3 如果还有下一 K-block，则把 reg_a_vec/reg_b_vec 写入共享内存 write_index ----
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = (i / a_load_stride) * 4;
                shared_a[write_index][OFFSET(a_load_col, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx];
                shared_a[write_index][OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 1];
                shared_a[write_index][OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 2];
                shared_a[write_index][OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] =
                    reg_a_vec[reg_idx + 3];
            }

#pragma unroll
            for (int i = 0; i < BLOCK_K; i += b_load_stride)
            {
                int reg_idx = (i / b_load_stride) * 4;
                FETCH_FLOAT4(shared_b[write_index][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                    FETCH_FLOAT4(reg_b_vec[reg_idx]);
            }

            __syncthreads();

            // ---- 11.4 从新写入的共享缓冲区中取出下一 K-block 的第 0 行到寄存器 ----
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[0][m]) =
                    FETCH_FLOAT4(shared_a[write_index][OFFSET(0, local_row_idx + m, BLOCK_M)]);
            }

#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[0][n]) =
                    FETCH_FLOAT4(shared_b[write_index][OFFSET(0, local_col_idx + n, BLOCK_N)]);
            }

            // 交替使用 0/1 缓冲区
            write_index ^= 1;
        }

        // ---- 11.5 处理当前 tile 的最后一个 k_inner = BLOCK_K-1 ----
#pragma unroll
        for (int m = 0; m < THREAD_M; m++)
        {
#pragma unroll
            for (int n = 0; n < THREAD_N; n++)
            {
                accum[m][n] += reg_a_tile[(BLOCK_K - 1) % 2][m] * reg_b_tile[(BLOCK_K - 1) % 2][n];
            }
        }

    } while (k < K);

    // ------------------------------
    // 12. 将累加结果写回全局内存 C
    //     执行 BLAS 规范中的：C = alpha * accum + beta * C
    //     这里使用 float4 向量化写回。
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

// 整除向上，用于根据 tile 尺寸计算 grid 大小
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// 生成一组测试矩阵尺寸（从 256 到 8192，步长 256）
std::vector<int> generateSizes()
{
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256)
        sizes.push_back(i);
    return sizes;
}

int main()
{
    // 生成一组要测试的矩阵尺寸
    std::vector<int> sizes = generateSizes();

    // 打开 CSV 文件，用于记录不同矩阵规模下的性能
    // 列含义：
    //   Size          : 矩阵边长 N
    //   CUBLAS_GFLOPS : cuBLAS 实测算力
    //   MySGEMM_FLOPS : 自实现 kernel 实测算力
    //   Matched       : 1=在 TOL 内与 cuBLAS 一致，0=存在明显差异
    //   Ratio         : MySGEMM_FLOPS / CUBLAS_GFLOPS
    std::ofstream csv_file("sgemm_benchmark_v4.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    // 遍历每一个矩阵规模进行测试
    for (int N : sizes)
    {
        std::cout << "Testing size " << N << std::endl;

        // 单个 N×N float 矩阵占用的字节数
        size_t size = static_cast<size_t>(N) * N * sizeof(float);

        // 在主机端分配内存
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size); // cuBLAS 结果
        float* host_C_v4 = (float*)malloc(size);     // 自实现 SGEMM 结果

        // 在 GPU 上分配内存
        float *device_a, *device_b, *device_c_v4;
        checkCudaError(cudaMalloc(&device_a, size), "cudaMalloc device_a failed");
        checkCudaError(cudaMalloc(&device_b, size), "cudaMalloc device_b failed");
        checkCudaError(cudaMalloc(&device_c_v4, size), "cudaMalloc device_c_v4 failed");

        bool out_of_memory = false;

        try
        {
            // 初始化主机矩阵：A 全 1，B 全 2，理论 C 元素 ≈ 2 * N
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // 把 A、B 复制到 GPU
            checkCudaError(cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_A to device failed");
            checkCudaError(cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_B to device failed");

            // 创建 cuBLAS 句柄
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f, beta = 0.0f;

            // 创建用于测量 GPU 时间的 event
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");

            // -------- 1) cuBLAS 预热，避免首次调用偏慢 -------- //
            int warmup_times = 10;
            for (int i = 0; i < warmup_times; i++)
            {
                // cuBLAS 语义上按列主序解释矩阵，这里输入是行主序；
                // 由于 A/B 全是常数矩阵，其乘积所有元素相等（2*N），
                // 因此这种“视图不一致”不会影响结果数值，用作对比是安全的。
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v4, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

            // -------- 2) 正式计时 cuBLAS -------- //
            int repeat_times = 50;
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v4, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize failed");

            // 计算 cuBLAS 总耗时（毫秒）
            float cublas_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime failed");

            // 取回 cuBLAS 结果，用于后续比较
            checkCudaError(cudaMemcpy(host_C_cublas, device_c_v4, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v4 to host_C_cublas failed");

            // 清空 C，准备自实现版本测试
            checkCudaError(cudaMemset(device_c_v4, 0, size), "cudaMemset device_c_v4 failed");

            // 配置自定义 kernel 的执行参数
            // BLOCK_M = BLOCK_N = 128, THREAD_M = THREAD_N = 8, BLOCK_K = 8
            // gridDim 以 C 的 tile 尺寸 128×128 为单元划分
            dim3 block(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            // -------- 3) 自定义 SGEMM 预热 -------- //
            for (int i = 0; i < warmup_times; i++)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, block>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v4);
            }
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

            // -------- 4) 正式计时自实现 SGEMM -------- //
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, block>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v4);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize failed");
            checkCudaError(cudaGetLastError(), "cudaGetLastError failed");

            float v4_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&v4_time, start, stop),
                           "cudaEventElapsedTime failed");

            // 取回自实现的 C 进行正确性验证
            checkCudaError(cudaMemcpy(host_C_v4, device_c_v4, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v4 to host_C_v4 failed");

            // 与 cuBLAS 结果逐元素比较，最多统计 10 个误差点
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_v4[i]) > TOL)
                {
                    error_count++;
                }
            }

            // 计算 GFLOPS（单次 GEMM ≈ 2*N^3 FLOPs）
            float cublas_gflops = (repeat_times * 2.0f * N * N * N) / (1e6f * cublas_time);
            float v4_gflops = (repeat_times * 2.0f * N * N * N) / (1e6f * v4_time);
            float ratio = v4_gflops / cublas_gflops;

            // 将结果写入 CSV：Matched 表示是否在 TOL 误差内与 cuBLAS 一致
            csv_file << N << "," << cublas_gflops << "," << v4_gflops << ","
                     << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

            // 回收资源
            cublasDestroy(handle);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaFree(device_a);
            cudaFree(device_b);
            cudaFree(device_c_v4);

            free(host_A);
            free(host_B);
            free(host_C_cublas);
            free(host_C_v4);

            cudaDeviceSynchronize();
        }
        catch (...)
        {
            // 如果发生异常（例如 OOM），记录并继续下一尺寸
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (!out_of_memory)
        {
            std::cout << "Finished size: " << N << std::endl;
        }
        else
        {
            // OOM 行补齐 Ratio 列，写 0 占位
            csv_file << N << ",OOM,OOM,0,0" << std::endl;
        }
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark_v4.csv'" << std::endl;

    return 0;
}
