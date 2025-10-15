#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024; // 1M elements
#define FULL_MASK 0xffffffff

// warp 级归约，针对 double 类型
__device__ double warpReduceSum(double val)
{
    // 每次将距离为 offset 的线程值相加，直到warp内只剩下一个结果
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(FULL_MASK, val, offset);
    return val;
}

// CPU版本用于验证
double reduce_cpu(const std::vector<double>& data)
{
    return std::accumulate(data.begin(), data.end(), 0.0);
}

// GPU版本：共享内存 + warp归约优化
__global__ void reduce_v3(const double* data_input, double* data_out)
{
    __shared__ double shared_data[BLOCK_SIZE]; // 每个block的共享内存

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    // 每个线程加载两个元素
    shared_data[tid] = data_input[i] + data_input[i + blockDim.x];
    __syncthreads();

    // block内的前半部分线程归约共享内存中的数据
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // 剩下最后一个warp时，用warp shuffle做最后的快速归约
    double val = shared_data[tid];
    if (tid < 32)
    {
        val = warpReduceSum(val);
    }

    // 将结果写回
    if (tid == 0)
    {
        data_out[blockIdx.x] = val;
    }
}

int main()
{
    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;

    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0); // 生成1~N的序列

    // ---- CPU计算 ----
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // ---- GPU计算 ----
    double *device_input, *device_output, *device_final_output;
    double gpu_result;

    cudaMalloc(&device_input, N * sizeof(double));
    cudaMalloc(&device_output, num_blocks * sizeof(double));
    cudaMalloc(&device_final_output, sizeof(double));

    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // 两轮归约
    reduce_v3<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output);
    reduce_v3<<<1, num_blocks>>>(device_output, device_final_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(&gpu_result, device_final_output, sizeof(double), cudaMemcpyDeviceToHost);

    // 验证结果
    if (abs(cpu_result - gpu_result) < 1e-6)
        std::cout << "GPU result verified SUCCESS" << std::endl;
    else
        std::cout << "GPU result verified FAILED" << std::endl;

    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_duration.count() / gpu_time << "x" << std::endl;

    // 清理资源
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_final_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
