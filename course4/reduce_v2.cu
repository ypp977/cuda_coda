#include <cuda_runtime.h>

#include <chrono> // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024; // 1M elements

__global__ void reduce_v2(const double* data_input, double* data_out)
{
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    shared_data[tid] = data_input[i] + data_input[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        data_out[blockIdx.x] = shared_data[0];
    }
}

double reduce_cpu(const std::vector<double>& data)
{
    return std::accumulate(data.begin(), data.end(), 0.0);
}

int main()
{
    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;

    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;

    double *device_input, *device_out, *device_final_out;
    double gpu_result;

    cudaMalloc(&device_input, N * sizeof(double));
    cudaMalloc(&device_out, num_blocks * sizeof(double));
    cudaMalloc(&device_final_out, sizeof(double));

    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduce_v2<<<num_blocks, BLOCK_SIZE>>>(device_input, device_out);

    reduce_v2<<<1, num_blocks>>>(device_out, device_final_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaMemcpy(&gpu_result, device_final_out, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    if (abs(cpu_result - gpu_result) < 1e-6)
    {
        std::cout << "Results verified success" << std::endl;
    }
    else
    {
        std::cout << "Results verified failed" << std::endl;
    }

    cudaFree(device_input);
    cudaFree(device_out);
    cudaFree(device_final_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
