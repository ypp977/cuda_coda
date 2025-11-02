#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // fabsf() 用于浮点数绝对值
#include <fstream> // 文件输入输出流，用于写 CSV
#include <iostream>
#include <vector>

#define TOL 1e-5f // 误差容忍阈值，用于验证矩阵计算结果

// 检查 CUDA API 调用错误
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查 cuBLAS API 调用错误
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
mysgemm_v2: 分块矩阵乘法（Tiled Matrix Multiplication）
计算公式: C = alpha * A * B + beta * C
------------------------------------------

参数说明：
    M: 矩阵 A 的行数
    N: 矩阵 B 的列数（也是矩阵 C 的列数）
    K: 矩阵 A 的列数 = 矩阵 B 的行数
    alpha: 矩阵乘法系数
    beta:  累加系数（控制是否叠加原有 C）
    A, B, C: 输入输出矩阵（行主序 Row-major）

矩阵维度：
    A: M × K
    B: K × N
    C: M × N

算法设计思路：
    1. 将矩阵按 tile（子块）分块，每个 block 计算 C 的一个 BLOCK_M × BLOCK_N 子矩阵。
    2. 每个线程在 tile 内负责一个 THREAD_M × THREAD_N 的局部计算。
    3. 使用共享内存缓存 A、B 的 tile，减少全局内存访问。
    4. 沿 K 方向逐块累积局部计算结果到寄存器 tmp。
    5. 最后将寄存器结果写回全局内存 C。
------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B,
                           float beta, float* C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // block_row, block_col 确定了 C_tile 在大矩阵中的位置
    // ------------------------------
    int block_col = blockIdx.x; // block 负责的列 tile
    int block_row = blockIdx.y; // block 负责的行 tile

    // ------------------------------
    // 2. Tile 内线程划分
    // threads_per_block_x/y 表示每个 tile 内线程块数
    // total_threads 是 tile 内总线程数
    // ------------------------------
    int threads_per_block_x = BLOCK_N / THREAD_N; // 横向线程数量
    int threads_per_block_y = BLOCK_M / THREAD_M; // 纵向线程数量
    int total_threads = threads_per_block_x * threads_per_block_y;

    // ------------------------------
    // 3. 当前线程在 tile 内负责的局部计算块起点
    // local_row/col 表示线程负责的 C_tile 局部子矩阵左上角
    // ------------------------------
    int local_col = (threadIdx.x % threads_per_block_x) * THREAD_N;
    int local_row = (threadIdx.x / threads_per_block_x) * THREAD_M;

    // ------------------------------
    // 4. 分配共享内存，用于缓存当前 tile 的 A、B 子块
    // shared_a: BLOCK_M × BLOCK_K
    // shared_b: BLOCK_K × BLOCK_N
    // ------------------------------
    __shared__ float shared_a[BLOCK_M * BLOCK_K];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算当前 block 对应的全局矩阵起始指针
    // ------------------------------
    A = &A[block_row * BLOCK_M * K];
    B = &B[block_col * BLOCK_N];
    C = &C[block_row * BLOCK_M * N + block_col * BLOCK_N];

    // ------------------------------
    // 6. 定义每个线程加载 A、B 子块时的分工
    // a_load_row/col、b_load_row/col 决定线程加载共享内存的位置
    // a_load_stride/b_load_stride 控制跨线程步长，保证 tile 全覆盖
    // ------------------------------
    int a_load_row = threadIdx.x / BLOCK_K;
    int a_load_col = threadIdx.x % BLOCK_K;
    int a_load_stride = total_threads / BLOCK_K;

    int b_load_row = threadIdx.x / BLOCK_N;
    int b_load_col = threadIdx.x % BLOCK_N;
    int b_load_stride = total_threads / BLOCK_N;

    // ------------------------------
    // 7. 局部寄存器缓存，用于保存线程负责的 C 子块结果
    // THREAD_M × THREAD_N 表示每个线程负责的计算块大小
    // ------------------------------
    float tmp[THREAD_M][THREAD_N] = {0.};

    // ------------------------------
    // 8. 主循环：沿 K 方向分块计算
    // 每次循环处理 BLOCK_K 深度的 A、B 子块
    // ------------------------------
#pragma unroll
    for (int k = 0; k < K; k += BLOCK_K)
    {
        // --------------------------
        // 8.1 从全局内存加载 A 子块到共享内存
        // 遍历 BLOCK_M 行，用 a_load_stride 跨线程步长避免重复加载
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            shared_a[(a_load_row + i) * BLOCK_K + a_load_col] =
                A[(a_load_row + i) * K + a_load_col];
        }

        // --------------------------
        // 8.2 从全局内存加载 B 子块到共享内存
        // 遍历 BLOCK_K 行，用 b_load_stride 跨线程步长避免重复加载
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            shared_b[(b_load_row + i) * BLOCK_N + b_load_col] =
                B[(b_load_row + i) * N + b_load_col];
        }

        // --------------------------
        // 8.3 同步线程，确保共享内存加载完毕
        // --------------------------
        __syncthreads();

        // --------------------------
        // 8.4 更新全局矩阵指针到下一个 K 子块位置
        // --------------------------
        A += BLOCK_K;     // A 向右移动 BLOCK_K 列
        B += BLOCK_K * N; // B 向下移动 BLOCK_K 行

        // --------------------------
        // 8.5 寄存器内乘加，累积 tmp
        // 每个线程计算 THREAD_M × THREAD_N 的结果
        // --------------------------
#pragma unroll
        for (int i = 0; i < BLOCK_K; i++)
        {
#pragma unroll
            for (int j = 0; j < THREAD_M; j++)
            {
                for (int l = 0; l < THREAD_N; l++)
                {
                    tmp[j][l] += shared_a[(local_row + j) * BLOCK_K + i] *
                                 shared_b[i * BLOCK_N + (local_col + l)];
                }
            }
        }
        __syncthreads(); // 等待所有线程完成本轮计算
    }

    // ------------------------------
    // 9. 将寄存器 tmp 中的结果写回全局内存 C
    // 注意考虑 alpha、beta 系数
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

// 向上取整函数，用于计算 grid size
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// 生成测试矩阵尺寸
std::vector<int> generateSizes()
{
    return {4096}; // 可以扩展多种尺寸测试
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
        // 分配主机内存
        // --------------------------
        float* host_a = (float*)malloc(size);
        float* host_b = (float*)malloc(size);
        float* host_c_cublas = (float*)malloc(size);
        float* host_c_v2 = (float*)malloc(size);

        // --------------------------
        // 分配设备内存
        // --------------------------
        float *device_a, *device_b, *device_c_v2;
        checkCudaError(cudaMalloc(&device_a, size), "cudaMalloc device_a failed");
        checkCudaError(cudaMalloc(&device_b, size), "cudaMalloc device_b failed");
        checkCudaError(cudaMalloc(&device_c_v2, size), "cudaMalloc device_c_v2 failed");

        bool out_of_memory = false;

        try
        {
            // --------------------------
            // 初始化矩阵 A、B
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
            // --------------------------
            // cuBLAS 预热
            // --------------------------
            for (int i = 0; i < warmup_times; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v2, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();
            checkCudaError(cudaMemset(device_c_v2, 0, size), "cudaMemset device_c_v2 failed");

            int repeat_times = 50;

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

            float cublas_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime failed");

            checkCudaError(cudaMemcpy(host_c_cublas, device_c_v2, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_c_cublas failed");

            checkCudaError(cudaMemset(device_c_v2, 0, size), "cudaMemset device_c_v2 failed");

            // --------------------------
            // MySGEMM 预热
            // --------------------------
            dim3 blockDim(256);
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

            float v2_time = 0.0f;
            checkCudaError(cudaEventElapsedTime(&v2_time, start, stop),
                           "cudaEventElapsedTime failed");

            checkCudaError(cudaMemcpy(host_c_v2, device_c_v2, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy host_c_v2 failed");

            // --------------------------
            // 检查计算结果
            // --------------------------
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_c_cublas[i] - host_c_v2[i]) > TOL)
                {
                    error_count++;
                }
            }

            float cublas_gflops = repeat_times * 2.0f * N * N * N / (cublas_time * 1e6f);
            float v2_gflops = repeat_times * 2.0f * N * N * N / (v2_time * 1e6f);

            csv_file << N << "," << cublas_gflops << "," << v2_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // --------------------------
            // 清理资源
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
