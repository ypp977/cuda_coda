#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024; // 每个线程块的线程数
const int N = 1024 * 1024;   // 总数据量（1M）

// GPU归约核函数：每个block对一部分数据求和
__global__ void reduce_v2(const double* data_input, double* data_out)
{
    __shared__ double shared_data[BLOCK_SIZE]; // block内共享内存，用于存储局部和

    int tid = threadIdx.x;                     // 线程索引（block内）
    int i = blockIdx.x * blockDim.x * 2 + tid; // 全局索引（每个block处理两倍于blockDim的数据）

    // 每个线程负责加两个元素，提高带宽利用率
    shared_data[tid] = data_input[i] + data_input[i + blockDim.x];
    __syncthreads(); // 等待所有线程加载完数据

    // 在共享内存中进行树形归约（从blockDim/2递减到1）
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s]; // 前一半线程加后一半线程的值
        }
        __syncthreads(); // 每轮迭代都要同步，防止读写冲突
    }

    // 每个block的第0号线程将局部结果写回全局内存
    if (tid == 0)
    {
        data_out[blockIdx.x] = shared_data[0];
    }
}

// CPU版本的归约，用于验证结果正确性
double reduce_cpu(const std::vector<double>& data)
{
    return std::accumulate(data.begin(), data.end(), 0.0);
}

int main()
{
    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2; // GPU第一轮归约所需的block数

    // 初始化输入数据（1.0, 2.0, 3.0, ...）
    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0);

    // ---- CPU计算 ----
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;

    // ---- GPU计算 ----
    double *device_input, *device_out, *device_final_out;
    double gpu_result;

    // 分配显存
    cudaMalloc(&device_input, N * sizeof(double));
    cudaMalloc(&device_out, num_blocks * sizeof(double));
    cudaMalloc(&device_final_out, sizeof(double));

    // 拷贝数据到GPU
    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // CUDA计时器
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 第一次归约：每个block求局部和
    reduce_v2<<<num_blocks, BLOCK_SIZE>>>(device_input, device_out);

    // 第二次归约：将上一次的结果再次归约到一个数
    reduce_v2<<<1, num_blocks>>>(device_out, device_final_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算GPU耗时
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // 拷贝结果回主机
    cudaMemcpy(&gpu_result, device_final_out, sizeof(double), cudaMemcpyDeviceToHost);

    // 输出结果与性能
    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    // 验证结果正确性
    if (abs(cpu_result - gpu_result) < 1e-6)
    {
        std::cout << "Results verified success" << std::endl;
    }
    else
    {
        std::cout << "Results verified failed" << std::endl;
    }

    // 释放资源
    cudaFree(device_input);
    cudaFree(device_out);
    cudaFree(device_final_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
