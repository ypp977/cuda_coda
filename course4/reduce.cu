#include <chrono> // 用于 CPU 计时
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

// 每个 block 的线程数
const int BLOCK_SIZE = 1024;
// 总数据量
const int N = 1024 * 1024; // 1M elements

// ----------------------- GPU Kernel: reduce_v0 -----------------------
__global__ void reduce_v0(double* input_data, double* out_data)
{
    // 共享内存，用于存储当前 block 内的数据，便于线程间快速访问
    __shared__ double shared_data[BLOCK_SIZE];

    // 当前线程在 block 内的索引
    int tid = threadIdx.x;
    // 当前线程在全局数据中的索引
    int i = blockIdx.x * blockDim.x + tid;

    // 将全局内存的数据拷贝到共享内存中，如果越界则置为0
    shared_data[tid] = (i < N) ? input_data[i] : 0.0;
    __syncthreads(); // 确保所有线程都完成拷贝

    // ---------------- 树状归约算法 ----------------
    // 每一轮将间隔为 s 的元素相加，s 从 1 开始，每轮翻倍
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        // 只有 tid 能整除 2*s 的线程执行加法
        // 例如 s=1 时：tid = 0,2,4,6...执行，加上右边相邻的元素
        if (tid % (2 * s) == 0)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        // 等待 block 内所有线程完成本轮加法
        __syncthreads();
    }

    // block 内的线程 0 保存该 block 的部分和到全局内存
    if (tid == 0)
    {
        out_data[blockIdx.x] = shared_data[0];
    }
}

// ----------------------- CPU 验证函数 -----------------------
double reduce_cpu(const std::vector<double>& input_data)
{
    double sum = 0.0;
    for (double val : input_data)
    {
        sum += val; // 顺序累加
    }
    return sum;
}

// ----------------------- main 函数 -----------------------
int main()
{
    // 计算需要多少个 block 来处理 N 个元素
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 初始化 host 数据，使用 std::iota 填充从 1.0 开始的连续值
    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0);

    // ---------------- CPU 计算 ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << ", time: " << cpu_duration.count() << " ms"
              << std::endl;
    std::cout << "GPU result: " << std::flush;

    // ---------------- GPU 内存分配 ----------------
    double *device_input, *device_out, *device_final_out;
    double gpu_result;

    // 输入数据
    cudaMalloc(&device_input, N * sizeof(double));
    // 输出数据，每个 block 的部分和
    cudaMalloc(&device_out, num_blocks * sizeof(double));
    // 最终结果（单个 double）
    cudaMalloc(&device_final_out, sizeof(double));

    // 将 host 数据拷贝到 GPU
    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ---------------- GPU 计时 ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ---------------- 第一次 kernel: 计算每个 block 的部分和 ----------------
    reduce_v0<<<num_blocks, BLOCK_SIZE>>>(device_input, device_out);
    // ---------------- 第二次 kernel: 将 block 部分和归约成最终结果 ----------------
    reduce_v0<<<1, num_blocks>>>(device_out, device_final_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_duration = 0.f;
    cudaEventElapsedTime(&gpu_duration, start, stop);

    // 拷贝 GPU 结果回 host
    cudaMemcpy(&gpu_result, device_final_out, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << gpu_result << ", time: " << gpu_duration << " ms" << std::endl;

    // ---------------- 验证结果 ----------------
    if (abs(cpu_result - gpu_result) < 1e-5)
    {
        std::cout << "Result verified successfully!" << std::endl;
    }
    else
    {
        std::cout << "Result verification failed!" << std::endl;
    }

    // ---------------- 释放 GPU 资源 ----------------
    cudaFree(device_input);
    cudaFree(device_out);
    cudaFree(device_final_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
