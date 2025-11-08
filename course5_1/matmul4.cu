#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf 用于浮点数比较
#include <fstream> // 文件输出流，用于写CSV
#include <iostream>
#include <vector>

// CUDA kernel 参数
#define BLOCK_SIZE 128 // 每个 block 的线程数
#define TOL 1e-5f      // 计算结果误差容忍度

// 检查 CUDA API 返回值，如果出错则打印信息并退出
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 返回值，如果出错则打印信息并退出
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 矩阵偏移宏，用于访问行列数据
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将连续的四个 float 视为 float4，方便向量化加载/存储
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

/*
------------------------------------------------------------
mysgemm_v4: 手写高性能SGEMM kernel (C = alpha * A * B + beta * C)
------------------------------------------------------------

模板参数:
    BM, BN, BK : Block级别tile大小
    TM, TN     : 每个线程负责的子块大小

函数参数:
    M, N, K : 矩阵维度 (A[MxK], B[KxN], C[MxN])
    alpha, beta : 缩放系数
    A, B      : 输入矩阵 (Row-major)
    C         : 输出矩阵

算法核心:
1. 将矩阵按tile分块，每个block处理C的BMxBN子矩阵。
2. 每个线程计算TMxTN的局部子矩阵。
3. 使用共享内存缓存A/B tile，减少全局内存访问。
4. 寄存器缓存线程加载的A/B数据，并累加到C子块。
5. 沿K方向逐块累积。
6. 最后将寄存器结果按alpha/beta写回全局内存。
------------------------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void __launch_bounds__(256)
    mysgemm_v4(int M, int N, int K, float alpha, float* __restrict__ A, float* __restrict__ B,
               float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // ------------------------------
    const int block_col_idx = blockIdx.x; // block 对应的列块索引
    const int block_row_idx = blockIdx.y; // block 对应的行块索引

    // ------------------------------
    // 2. Tile 内线程划分
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;                // 每行负责的线程数
    const int thread_per_col = BLOCK_M / THREAD_M;                // 每列负责的线程数
    const int thread_per_block = thread_per_row * thread_per_col; // block 内线程总数

    // ------------------------------
    // 3. 当前线程在 tile 内负责计算的局部 C 子块起点
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N; // C 子块列起点
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M; // C 子块行起点

    // ------------------------------
    // 4. 分配共享内存用于缓存 A/B tile (双缓冲)
    // shared_a: BLOCK_K x BLOCK_M, 转置存储以便 coalesced 访存
    // shared_b: BLOCK_K x BLOCK_N, 按原始布局存储
    // ------------------------------
    __shared__ float shared_a[2][BLOCK_K * BLOCK_M];
    __shared__ float shared_b[2][BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 每个线程需要加载的 float4 向量数，用于向量化加载
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 计算线程加载 A/B 的行列起点和步长
    // ------------------------------
    const int a_load_row = threadIdx.x / (BLOCK_K / 4);
    const int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int a_load_stride = BLOCK_M / vec4_load_per_thread_a;

    const int b_load_row = threadIdx.x / (BLOCK_N / 4);
    const int b_load_col = (threadIdx.x % (BLOCK_N / 4)) * 4;
    const int b_load_stride = BLOCK_N / vec4_load_per_thread_b;

    // ------------------------------
    // 7. 寄存器缓存
    // accum: 存放每个线程计算的 C 子块
    // reg_a_vec/reg_b_vec: 线程向量化加载的 A/B 数据
    // reg_a_tile/reg_b_tile: 当前迭代的共享内存 tile 缓存
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.};

    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.};
    float reg_b_vec[4 * vec4_load_per_thread_b] = {0.};

    float reg_a_tile[2][THREAD_M];
    float reg_b_tile[2][THREAD_N];

    // ------------------------------
    // 8. 将 A/B/C 指针偏移到当前 block 对应子矩阵
    // ------------------------------
    A = &A[block_row_idx * BLOCK_M * K];                           // A 的行偏移
    B = &B[block_col_idx * BLOCK_N];                               // B 的列偏移
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N]; // C 的 block 起点

    // ------------------------------
    // 9. 将 A/B tile 加载到共享内存 (双缓冲初始化)
    // ------------------------------
#pragma unroll
    for (int i = 0; i < BLOCK_M; i += a_load_stride)
    {
        int reg_idx = i / a_load_stride * 4;
        FETCH_FLOAT4(reg_a_vec[reg_idx]) = FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

        shared_a[0][OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
        shared_a[0][OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
        shared_a[0][OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
        shared_a[0][OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
    }

#pragma unroll
    for (int i = 0; i < BLOCK_K; i += b_load_stride)
    {
        FETCH_FLOAT4(shared_b[0][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
            FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
    }
    __syncthreads();

    // ------------------------------
    // 10. 寄存器 tile 初始化 (从共享内存加载到寄存器)
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
    // 11. K 方向主循环 (双缓冲，预加载下一 tile)
    // ------------------------------
    int write_index = 1; // 双缓冲索引
    int load_index;
    int k = 0;

    do
    {
        k += BLOCK_K;

        if (k < K)
        {
            // ---- 11.1 线程加载下一 A/B tile 到寄存器
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = i / a_load_stride * 4;
                FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                    FETCH_FLOAT4(A[OFFSET(a_load_row + i, k + a_load_col, K)]);
            }

#pragma unroll
            for (int i = 0; i < BLOCK_K; i += b_load_stride)
            {
                int reg_idx = i / b_load_stride * 4;
                FETCH_FLOAT4(reg_b_vec[reg_idx]) =
                    FETCH_FLOAT4(B[OFFSET(k + b_load_row + i, b_load_col, N)]);
            }
        }

        load_index = write_index ^ 1; // 切换共享内存读取索引

        // ---- 11.2 在寄存器中计算当前 C 子块累加 ----
#pragma unroll
        for (int bk = 0; bk < BLOCK_K - 1; bk++)
        {
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[(bk + 1) % 2][m]) =
                    FETCH_FLOAT4(shared_a[load_index][OFFSET(bk + 1, local_row_idx + m, BLOCK_M)]);
            }

#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[(bk + 1) % 2][n]) =
                    FETCH_FLOAT4(shared_b[load_index][OFFSET(bk + 1, local_col_idx + n, BLOCK_N)]);
            }

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

        // ---- 11.3 将下一 tile 写入共享内存，切换双缓冲索引 ----
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BLOCK_M; i += a_load_stride)
            {
                int reg_idx = i / a_load_stride * 4;
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
                int reg_idx = i / b_load_stride * 4;
                FETCH_FLOAT4(shared_b[write_index][OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                    FETCH_FLOAT4(reg_b_vec[reg_idx]);
            }
            __syncthreads();

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

            write_index ^= 1; // 切换双缓冲索引
        }

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
    // 12. 将累加结果写回全局内存
    // 按 BLAS 规范执行 alpha * A*B + beta * C
    // 使用 float4 向量化写回
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

// 整除向上宏
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// 生成测试矩阵大小
std::vector<int> generateSizes()
{
    std::vector<int> sizes;
    for (int i = 256; i <= 8192; i += 256)
        sizes.push_back(i);
    return sizes;
}

int main()
{
    // ------------------------------
    // 1. 生成测试矩阵尺寸列表
    // generateSizes() 返回一个 vector<int>，包含待测试的 N（矩阵大小 N x N）
    // ------------------------------
    std::vector<int> sizes = generateSizes();

    // ------------------------------
    // 2. 打开 CSV 文件，用于保存性能测试结果
    // 列名包括矩阵大小、cuBLAS GFLOPS、手写 kernel GFLOPS、是否匹配、性能比值
    // ------------------------------
    std::ofstream csv_file("sgemm_benchmark_v4.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    // ------------------------------
    // 3. 遍历每个测试矩阵大小
    // ------------------------------
    for (int N : sizes)
    {
        std::cout << "Testing size " << N << std::endl;

        size_t size = N * N * sizeof(float); // 计算矩阵总字节数

        // ------------------------------
        // 3.1 CPU 内存分配
        // host_A、host_B：输入矩阵
        // host_C_cublas：cuBLAS 输出结果
        // host_C_v4：手写 kernel 输出结果
        // ------------------------------
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_v4 = (float*)malloc(size);

        // ------------------------------
        // 3.2 GPU 内存分配
        // device_a、device_b：输入矩阵
        // device_c_v4：输出矩阵
        // ------------------------------
        float *device_a, *device_b, *device_c_v4;
        checkCudaError(cudaMalloc(&device_a, size), "cudaMalloc device_a  failed");
        checkCudaError(cudaMalloc(&device_b, size), "cudaMalloc device_b  failed");
        checkCudaError(cudaMalloc(&device_c_v4, size), "cudaMalloc device_c_v4 failed");

        bool out_of_memory = false; // 标记是否发生 OOM 或其他异常

        try
        {
            // ------------------------------
            // 4. 初始化矩阵数据
            // 便于验证结果正确性，这里都用固定值
            // ------------------------------
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // ------------------------------
            // 5. 将矩阵拷贝到 GPU
            // ------------------------------
            checkCudaError(cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_A to device failed");
            checkCudaError(cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy host_B to device failed");

            // ------------------------------
            // 6. 创建 cuBLAS handle
            // ------------------------------
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f, beta = 0.0f; // SGEMM 参数 alpha, beta

            // ------------------------------
            // 7. 创建 CUDA 事件，用于计时
            // ------------------------------
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate start failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop failed");

            // ------------------------------
            // 8. cuBLAS warmup
            // 多次执行以稳定 GPU 性能
            // ------------------------------
            int warpup_times = 10;
            for (int i = 0; i < warpup_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v4, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

            // ------------------------------
            // 9. cuBLAS 性能测试
            // ------------------------------
            int repeat_times = 50; // 重复次数，用于平均性能
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v4, N),
                                 "cublasSgemm failed");
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime failed");

            // ------------------------------
            // 10. 拷贝 cuBLAS 结果回 CPU
            // ------------------------------
            checkCudaError(cudaMemcpy(host_C_cublas, device_c_v4, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v4 to host failed");

            // ------------------------------
            // 11. 清零输出矩阵，为手写 kernel 做准备
            // ------------------------------
            checkCudaError(cudaMemset(device_c_v4, 0, size), "cudaMemset device_c_v4 failed");

            // ------------------------------
            // 12. 设置 kernel 配置
            // block：每个 block 的线程数
            // gridDim：计算 grid 的大小
            // ------------------------------
            dim3 block(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            // ------------------------------
            // 13. 手写 kernel warmup
            // ------------------------------
            for (int i = 0; i < warpup_times; i++)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, block>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v4);
            }
            checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

            // ------------------------------
            // 14. 手写 kernel 性能测试
            // ------------------------------
            checkCudaError(cudaEventRecord(start), "cudaEventRecord start failed");
            for (int i = 0; i < repeat_times; i++)
            {
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, block>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v4);
            }
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop failed");
            checkCudaError(cudaGetLastError(), "cudaGetLastError failed");

            float v4_time = 0;
            checkCudaError(cudaEventElapsedTime(&v4_time, start, stop),
                           "cudaEventElapsedTime failed");

            // ------------------------------
            // 15. 拷贝手写 kernel 结果回 CPU
            // ------------------------------
            checkCudaError(cudaMemcpy(host_C_v4, device_c_v4, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy device_c_v4 to host failed");

            // ------------------------------
            // 16. 检查结果匹配性
            // TOL 为容忍误差
            // ------------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_v4[i]) > TOL)
                {
                    error_count++;
                }
            }

            // ------------------------------
            // 17. 计算 GFLOPS
            // ------------------------------
            float cublas_gflops = (repeat_times * 2.0f * N * N * N) / (1e6f * cublas_time);
            float v4_gflops = (repeat_times * 2.0f * N * N * N) / (1e6f * v4_time);
            float ratio = v4_gflops / cublas_gflops;

            // ------------------------------
            // 18. 写入 CSV 文件
            // Matched: 1 表示结果一致，0 表示不一致
            // ------------------------------
            csv_file << N << "," << cublas_gflops << "," << v4_gflops << ","
                     << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

            // ------------------------------
            // 19. 释放资源
            // ------------------------------
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
            // 捕获异常（如 OOM）并记录
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        // ------------------------------
        // 20. 输出当前测试状态
        // ------------------------------
        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    // ------------------------------
    // 21. 关闭 CSV 文件
    // ------------------------------
    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmarkV4.csv'" << std::endl;

    return 0;
}
