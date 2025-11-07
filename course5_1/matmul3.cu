#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>   // for fabsf
#include <fstream> // for CSV output
#include <iostream>
#include <vector>

// 定义比较精度容忍度
#define TOL 1e-5f
// 计算二维数组中元素的线性索引: row * leading_dimension + col
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 将指针转换为float4类型进行向量化内存访问
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 检查CUDA运行时API调用是否出错，并打印相关信息
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        // 修改为更准确的错误信息格式
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// 检查cuBLAS库函数调用是否出错，并打印相关信息
void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        // 修改为更准确的错误信息格式
        std::cerr << "CUBLAS Error: " << msg << " - Status code: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

/*
------------------------------------------
mysgemm_v3: 高性能手写矩阵乘法 kernel（C = alpha * A * B + beta * C）
------------------------------------------

模板参数：
    BLOCK_M, BLOCK_N, BLOCK_K : 每个 thread block 处理的子矩阵大小
    THREAD_M, THREAD_N        : 每个线程计算的输出子块大小

函数参数：
    M, N, K       : 矩阵维度 (A[M×K], B[K×N], C[M×N])
    alpha, beta   : 缩放系数（与 BLAS 规范一致）
    A, B          : 输入矩阵（行主序 Row-major）
    C             : 输出矩阵

算法设计思路：
    1. 将矩阵按 tile（子块）分块，每个 block 计算 C 的 BLOCK_M × BLOCK_N 子矩阵。
    2. 每个线程在 tile 内负责一个 THREAD_M × THREAD_N 的局部计算。
    3. 使用共享内存缓存 A、B 的 tile，减少全局内存访问。
    4. 使用寄存器缓存线程加载的 A/B 数据，并在寄存器中累加计算结果。
    5. 沿 K 方向逐块累积局部计算结果到寄存器 accum。
    6. 最后将寄存器结果按 alpha、beta 系数写回全局内存 C。
------------------------------------------
*/
template <const int BLOCK_M, const int BLOCK_N, const int BLOCK_K, const int THREAD_M,
          const int THREAD_N>
__global__ void mysgemm_v3(int M, int N, int K, float alpha, float* __restrict__ A,
                           float* __restrict__ B, float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前 block 在 C 矩阵中的 tile 坐标
    // block_row_idx: block 在 C 中的行编号
    // block_col_idx: block 在 C 中的列编号
    // ------------------------------
    const int block_col_idx = blockIdx.x;
    const int block_row_idx = blockIdx.y;

    // ------------------------------
    // 2. Tile 内线程划分
    // thread_per_row/col: tile 内线程数量
    // thread_per_block: tile 内总线程数量
    // ------------------------------
    const int thread_per_row = BLOCK_N / THREAD_N;
    const int thread_per_col = BLOCK_M / THREAD_M;
    const int thread_per_block = thread_per_row * thread_per_col;

    // ------------------------------
    // 3. 当前线程在 tile 内负责计算的局部 C 子块起点（左上角坐标）
    // local_row_idx / local_col_idx 表示线程在 tile 内的局部偏移
    // ------------------------------
    const int local_col_idx = (threadIdx.x % thread_per_row) * THREAD_N;
    const int local_row_idx = (threadIdx.x / thread_per_row) * THREAD_M;

    // ------------------------------
    // 4. 分配共享内存用于缓存当前 tile 的 A/B 子块
    // shared_a: BLOCK_K × BLOCK_M，用于缓存 A tile 并可转置访问
    // shared_b: BLOCK_K × BLOCK_N，用于缓存 B tile
    // ------------------------------
    __shared__ float shared_a[BLOCK_K * BLOCK_M];
    __shared__ float shared_b[BLOCK_K * BLOCK_N];

    // ------------------------------
    // 5. 计算每个线程需要加载的 float4 向量数
    // 向量化加载可以减少全局内存访问次数
    // ------------------------------
    const int vec4_load_per_thread_a = BLOCK_K * BLOCK_M / thread_per_block / 4;
    const int vec4_load_per_thread_b = BLOCK_K * BLOCK_N / thread_per_block / 4;

    // ------------------------------
    // 6. 计算线程加载 A/B 的行列起点和步长
    // a_load_row/col, b_load_row/col: 当前线程在 tile 内加载起点
    // a_load_stride/b_load_stride: 跨线程加载步长，保证 tile 全覆盖
    // ------------------------------
    const int a_load_row = threadIdx.x / (BLOCK_K / 4);
    const int a_load_col = (threadIdx.x % (BLOCK_K / 4)) * 4;
    const int a_load_stride = BLOCK_M / vec4_load_per_thread_a;

    const int b_load_row = threadIdx.x / (BLOCK_N / 4);
    const int b_load_col = (threadIdx.x % (BLOCK_N / 4)) * 4;
    const int b_load_stride = BLOCK_K / vec4_load_per_thread_b;

    // ------------------------------
    // 7. 寄存器缓存
    // accum: 存放每个线程计算的 C 子块结果
    // reg_a_vec: 缓存线程加载的 A 向量
    // reg_a_tile / reg_b_tile: 当前迭代从共享内存取出的 A/B 元素
    // ------------------------------
    float accum[THREAD_M][THREAD_N] = {0.0f};
    float reg_a_vec[4 * vec4_load_per_thread_a] = {0.};
    float reg_a_tile[THREAD_M];
    float reg_b_tile[THREAD_N];

    // ------------------------------
    // 8. 将 A/B/C 指针偏移到当前 block 对应的子矩阵起始位置
    // ------------------------------
    A = &A[block_row_idx * BLOCK_M * K];                           // 当前 block A 子块起始行
    B = &B[block_col_idx * BLOCK_N];                               // 当前 block B 子块起始列
    C = &C[block_row_idx * BLOCK_M * N + block_col_idx * BLOCK_N]; // 输出子块起始位置

    // ------------------------------
    // 9. 沿 K 方向循环处理 BLOCK_K 大小的 tile
    // ------------------------------
#pragma unroll
    for (int k_block_start = 0; k_block_start < K; k_block_start += BLOCK_K)
    {
        // ---- 9.1 将 A tile 从全局内存加载到共享内存并转置 ----
#pragma unroll
        for (int i = 0; i < BLOCK_M; i += a_load_stride)
        {
            const int reg_idx = i / a_load_stride * 4;
            FETCH_FLOAT4(reg_a_vec[reg_idx]) =
                FETCH_FLOAT4(A[OFFSET(a_load_row + i, a_load_col, K)]);

            // 将 A tile 转置存入共享内存，便于连续访存
            shared_a[OFFSET(a_load_col, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx];
            shared_a[OFFSET(a_load_col + 1, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 1];
            shared_a[OFFSET(a_load_col + 2, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 2];
            shared_a[OFFSET(a_load_col + 3, i + a_load_row, BLOCK_M)] = reg_a_vec[reg_idx + 3];
        }

        // ---- 9.2 将 B tile 从全局内存加载到共享内存 ----
#pragma unroll
        for (int i = 0; i < BLOCK_K; i += b_load_stride)
        {
            FETCH_FLOAT4(shared_b[OFFSET(b_load_row + i, b_load_col, BLOCK_N)]) =
                FETCH_FLOAT4(B[OFFSET(b_load_row + i, b_load_col, N)]);
        }

        __syncthreads(); // 保证所有线程完成共享内存加载

        // ---- 9.3 移动全局指针到下一 K 块 ----
        A += BLOCK_K;     // A 往右移动 BLOCK_K 列
        B += BLOCK_K * N; // B 往下移动 BLOCK_K 行

        // ---- 9.4 在共享内存中执行矩阵乘法 ----
#pragma unroll
        for (int k_inner = 0; k_inner < BLOCK_K; k_inner++)
        {
            // 加载 A tile 当前行到寄存器
#pragma unroll
            for (int m = 0; m < THREAD_M; m += 4)
            {
                FETCH_FLOAT4(reg_a_tile[m]) =
                    FETCH_FLOAT4(shared_a[OFFSET(k_inner, local_row_idx + m, BLOCK_M)]);
            }

            // 加载 B tile 当前列到寄存器
#pragma unroll
            for (int n = 0; n < THREAD_N; n += 4)
            {
                FETCH_FLOAT4(reg_b_tile[n]) =
                    FETCH_FLOAT4(shared_b[OFFSET(k_inner, local_col_idx + n, BLOCK_N)]);
            }

            // 在寄存器中累加 C 子块
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
        __syncthreads(); // 保证累加完成后再进行下一 K tile
    }

    // ------------------------------
    // 10. 将累加结果写回全局内存
    // ------------------------------
#pragma unroll
    for (int m = 0; m < THREAD_M; m++)
    {
#pragma unroll
        for (int n = 0; n < THREAD_N; n += 4)
        {
            float4 c_val = FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]);
            // 按 BLAS 规范计算 alpha * A*B + beta * C
            c_val.x = alpha * accum[m][n] + beta * c_val.x;
            c_val.y = alpha * accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * accum[m][n + 3] + beta * c_val.w;
            FETCH_FLOAT4(C[OFFSET(local_row_idx + m, local_col_idx + n, N)]) = c_val;
        }
    }
}

// 计算向上取整的除法
#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)

// 生成测试矩阵大小
std::vector<int> generateSizes()
{
    return {4096};
}

// 主函数：性能测试和正确性验证
int main()
{
    int device_id = 0;
    checkCudaError(cudaSetDevice(device_id), "Failed to set CUDA device");
    std::vector<int> sizes = generateSizes();

    // 创建CSV文件用于记录测试结果
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

    // 对不同大小的矩阵进行测试
    for (int N : sizes)
    {
        std::cout << "Testing Size: " << N << std::endl;

        // 计算矩阵所需内存大小
        size_t size = N * N * sizeof(float);

        // 分配主机端内存
        float* host_A = (float*)malloc(size);
        float* host_B = (float*)malloc(size);
        float* host_C_cublas = (float*)malloc(size);
        float* host_C_v3 = (float*)malloc(size);

        // 分配设备端内存
        float *device_a, *device_b, *device_c_v3;
        checkCudaError(cudaMalloc(&device_a, size),
                       "Failed to allocate device memory for matrix A");
        checkCudaError(cudaMalloc(&device_b, size),
                       "Failed to allocate device memory for matrix B");
        checkCudaError(cudaMalloc(&device_c_v3, size),
                       "Failed to allocate device memory for matrix C");

        try
        {
            // 初始化输入矩阵A和B
            for (int i = 0; i < N * N; i++)
            {
                host_A[i] = 1.0f;
                host_B[i] = 2.0f;
            }

            // 将输入矩阵从主机复制到设备
            checkCudaError(cudaMemcpy(device_a, host_A, size, cudaMemcpyHostToDevice),
                           "Failed to copy matrix A from host to device");
            checkCudaError(cudaMemcpy(device_b, host_B, size, cudaMemcpyHostToDevice),
                           "Failed to copy matrix B from host to device");

            // 创建cuBLAS句柄
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

            // 设置SGEMM参数
            float alpha = 1.0f, beta = 0.0f;

            // 创建CUDA事件用于计时
            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "Failed to create CUDA start event");
            checkCudaError(cudaEventCreate(&stop), "Failed to create CUDA stop event");

            // 预热运行cuBLAS SGEMM
            int warmip_time = 10;
            for (int i = 0; i < warmip_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v3, N),
                                 "Failed to execute cuBLAS SGEMM operation");
            }
            cudaDeviceSynchronize();

            // 正式测试cuBLAS SGEMM性能
            int repeat_time = 50;
            checkCudaError(cudaEventRecord(start), "Failed to record start event");
            for (int i = 0; i < repeat_time; i++)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                                             device_a, N, device_b, N, &beta, device_c_v3, N),
                                 "Failed to execute cuBLAS SGEMM operation");
            }
            checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
            checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

            // 计算cuBLAS SGEMM执行时间
            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "Failed to calculate elapsed time");

            // 将cuBLAS结果复制回主机
            checkCudaError(cudaMemcpy(host_C_cublas, device_c_v3, size, cudaMemcpyDeviceToHost),
                           "Failed to copy result matrix from device to host");

            // 清零输出矩阵，为自定义SGEMM做准备
            checkCudaError(cudaMemset(device_c_v3, 0, size),
                           "Failed to initialize matrix C with zeros");

            // 设置自定义SGEMM的执行配置
            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            // 预热运行自定义SGEMM
            for (int i = 0; i < warmip_time; i++)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v3);
            }
            cudaDeviceSynchronize();

            // 正式测试自定义SGEMM性能
            checkCudaError(cudaEventRecord(start), "Failed to record start event for custom SGEMM");

            for (int i = 0; i < repeat_time; i++)
            {
                mysgemm_v3<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, device_a, device_b, beta, device_c_v3);
            }
            checkCudaError(cudaEventRecord(stop), "Failed to record stop event for custom SGEMM");
            checkCudaError(cudaEventSynchronize(stop),
                           "Failed to synchronize stop event for custom SGEMM");

            // 计算自定义SGEMM执行时间
            float v3_time = 0;
            checkCudaError(cudaEventElapsedTime(&v3_time, start, stop),
                           "Failed to calculate elapsed time for custom SGEMM");

            // 将自定义SGEMM结果复制回主机
            checkCudaError(cudaMemcpy(host_C_v3, device_c_v3, size, cudaMemcpyDeviceToHost),
                           "Failed to copy result matrix from device to host");

            // 比较cuBLAS和自定义SGEMM的结果是否一致
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; i++)
            {
                if (fabsf(host_C_cublas[i] - host_C_v3[i]) > TOL)
                {
                    error_count++;
                }
            }

            // 计算GFLOPS性能指标
            float cublas_gflops = (repeat_time * 2.0f * N * N * N) / (cublas_time * 1e6f);
            float v3_gflops = (repeat_time * 2.0f * N * N * N) / (v3_time * 1e6f);

            // 将结果写入CSV文件
            csv_file << N << "," << cublas_gflops << "," << v3_gflops << ","
                     << (error_count == 0 ? "1" : "0") << std::endl;

            // 清理资源
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
