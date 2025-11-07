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
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    mysgemm_v4(int M, int N, int K, float alpha, float* __restrict__ A, float* __restrict__ B,
               float beta, float* __restrict__ C)
{
    // ------------------------------
    // 1. 当前Block在C矩阵中的位置
    // ------------------------------
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // ------------------------------
    // 2. Tile内线程划分
    // ------------------------------
    const int threads_per_row = BN / TN;
    const int threads_per_col = BM / TM;
    const int thread_num = threads_per_row * threads_per_col;

    // ------------------------------
    // 3. 当前线程负责的局部C子块左上角
    // ------------------------------
    int tx = (threadIdx.x % threads_per_row) * TN;
    int ty = (threadIdx.x / threads_per_row) * TM;

    // ------------------------------
    // 4. 分配共享内存存储A/B tile
    // double buffer设计: 0/1交替加载
    // ------------------------------
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    // ------------------------------
    // 5. 线程向量化加载数量
    // ------------------------------
    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    // ------------------------------
    // 6. 线程加载tile起点/步长
    // ------------------------------
    int a_tile_row = threadIdx.x / (BK / 4);
    int a_tile_col = (threadIdx.x % (BK / 4)) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN / 4);
    int b_tile_col = (threadIdx.x % (BN / 4)) * 4;
    int b_tile_stride = BK / ldg_b_num;

    // ------------------------------
    // 7. 寄存器缓存
    // accum: 线程累加结果
    // ldg_a_reg / ldg_b_reg: 线程加载A/B向量缓存
    // a_frag / b_frag: 当前迭代共享内存tile缓存
    // ------------------------------
    float accum[TM][TN] = {0.};
    float ldg_a_reg[4 * ldg_a_num] = {0.};
    float ldg_b_reg[4 * ldg_b_num] = {0.};
    float a_frag[2][TM];
    float b_frag[2][TN];

    // ------------------------------
    // 8. 指针偏移到当前Block对应子矩阵
    // ------------------------------
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // ------------------------------
    // 9. 预加载第一个tile到共享内存
    // ------------------------------
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride)
    {
        int ldg_index = i / a_tile_stride * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // 存入共享内存
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride)
    {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
            FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    __syncthreads();

    // ------------------------------
    // 10. 预加载到寄存器进行首轮计算
    // ------------------------------
#pragma unroll
    for (int m = 0; m < TM; m += 4)
    {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]);
    }
#pragma unroll
    for (int n = 0; n < TN; n += 4)
    {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]);
    }

    int write_index = 1;
    int load_index;
    int k = 0;
    do
    {
        k += BK;
        if (k < K)
        {
            // ---- 加载下一个A/B tile到寄存器
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride)
            {
                int ldg_index = i / a_tile_stride * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride)
            {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                    FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }

        load_index = write_index ^ 1;

        // ------------------------------
        // 11. 核心计算循环
        // ------------------------------
#pragma unroll
        for (int bk = 0; bk < BK - 1; bk++)
        {
#pragma unroll
            for (int m = 0; m < TM; m++)
            {
#pragma unroll
                for (int n = 0; n < TN; n++)
                {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }

        // ------------------------------
        // 12. 双缓冲写回共享内存
        // ------------------------------
        if (k < K)
        {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride)
            {
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] =
                    ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] =
                    ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] =
                    ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride)
            {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
            write_index ^= 1;
        }

    } while (k < K);

    // ------------------------------
    // 13. 将累加结果写回全局内存
    // ------------------------------
#pragma unroll
    for (int m = 0; m < TM; m++)
    {
#pragma unroll
        for (int n = 0; n < TN; n += 4)
        {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
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
    std::vector<int> sizes = generateSizes();

    // 打开CSV文件用于记录性能测试结果
    std::ofstream csv_file("sgemm_benchmark_v7.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched,Ratio" << std::endl;

    for (int N : sizes)
    {
        std::cout << "Testing size: " << N << std::endl;

        size_t size = N * N * sizeof(float);

        // CPU 分配内存
        float* A = (float*)malloc(size);
        float* B = (float*)malloc(size);
        float* C_cublas = (float*)malloc(size);
        float* C_v1 = (float*)malloc(size);

        // GPU 分配内存
        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

        bool out_of_memory = false;

        try
        {
            // 初始化矩阵 A 和 B
            for (int i = 0; i < N * N; ++i)
            {
                A[i] = 1.0f;
                B[i] = 2.0f;
            }

            // 拷贝数据到 GPU
            checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy A to device failed");
            checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                           "cudaMemcpy B to device failed");

            // cuBLAS handle 创建
            cublasHandle_t handle;
            checkCublasError(cublasCreate(&handle), "cublasCreate failed");

            float alpha = 1.0f;
            float beta = 0.0f;

            cudaEvent_t start, stop;
            checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
            checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

            // warmup 先执行几次，稳定 GPU 性能
            int warpup_time = 10;
            for (int i = 0; i < warpup_time; ++i)
            {
                checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B,
                                             N, d_A, N, &beta, d_C_v1, N),
                                 "cublasSgemm failed");
            }
            cudaDeviceSynchronize();

            // cuBLAS SGEMM 性能测试
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

            float cublas_time = 0;
            checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                           "cudaEventElapsedTime cublas failed");

            // 拷贝 cuBLAS 结果回 CPU
            checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_cublas failed");

            // 手写 kernel 测试
            checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

            dim3 blockDim(256);
            dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

            for (int i = 0; i < warpup_time; ++i)
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            cudaDeviceSynchronize();

            checkCudaError(cudaEventRecord(start), "cudaEventRecord(start v1) failed");
            for (int i = 0; i < repeat_time; ++i)
                mysgemm_v4<128, 128, 8, 8, 8>
                    <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
            checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
            checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize v1 failed");
            checkCudaError(cudaGetLastError(), "cuda get last error failed");

            float v1_time = 0;
            checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                           "cudaEventElapsedTime v1 failed");

            // 拷贝手写 kernel 结果回 CPU
            checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                           "cudaMemcpy C_v1 failed");

            // 检查结果是否匹配
            int error_count = 0;
            for (int i = 0; i < N * N && error_count < 10; ++i)
            {
                if (fabsf(C_cublas[i] - C_v1[i]) > TOL)
                    error_count++;
            }

            // 计算 GFLOPS
            float cublas_gflops = repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f); // GFlops
            float v1_gflops = repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);         // GFlops

            float ratio = v1_gflops / cublas_gflops;

            // 写入 CSV
            csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
                     << (error_count == 0 ? "1" : "0") << "," << ratio << std::endl;

            // 释放资源
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
            std::cerr << "Out of memory or error during testing size: " << N << std::endl;
            out_of_memory = true;
        }

        if (!out_of_memory)
            std::cout << "Finished size: " << N << std::endl;
        else
            csv_file << N << ",OOM,OOM,0" << std::endl;
    }

    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'" << std::endl;
    return 0;
}
