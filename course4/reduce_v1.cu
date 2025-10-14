#include <chrono> // 用于 CPU 计时
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024; // 每个 block 中的线程数
const int N = 1024 * 1024;   // 总数据量 (1M 个元素)

// ----------------------- GPU 归约核函数 v1 -----------------------
__global__ void reduce_v1(double* data_input, double* data_out)
{
    // 为当前 block 分配共享内存，用于临时存放每个线程的部分和
    __shared__ double temp[BLOCK_SIZE];

    int tid = threadIdx.x; // 当前线程在 block 内的索引
    // 每个线程负责加载两个元素：一个在前半区，一个在后半区
    int i = blockIdx.x * (blockDim.x * 2) + tid;

    // 先将两个元素相加，减少一半的全局内存访问次数
    temp[tid] = data_input[i] + data_input[i + blockDim.x];
    __syncthreads(); // 确保所有线程完成加载和加法操作

    // ---------------- 树状归约（树形加法） ----------------
    // 每轮把间隔为 s 的线程对其右侧的值累加上去
    // 例如 s=1：0加1, 2加3, 4加5 ...
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            temp[tid] += temp[tid + s];
        }
        __syncthreads(); // 确保一轮的所有加法都完成
    }

    // block 内线程 0 保存当前 block 的总和到全局内存
    if (tid == 0)
    {
        data_out[blockIdx.x] = temp[0];
    }
}

// ----------------------- CPU 归约函数（验证用） -----------------------
double reduce_cpu(std::vector<double>& data)
{
    // 使用 STL 提供的累加函数
    return std::accumulate(data.begin(), data.end(), 0.0);
}

// ----------------------- 主函数 -----------------------
int main()
{
    // 每个 block 处理 2 * BLOCK_SIZE 个数据（因为每个线程加载两个数）
    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;

    // 初始化 host 数据
    std::vector<double> host_data(N);
    for (int i = 0; i < N; i++)
    {
        host_data[i] = 1.0; // 所有值都为 1.0，理论总和 = N
    }

    // ---------------- CPU 部分 ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // ---------------- GPU 内存准备 ----------------
    double *device_input, *device_output, *device_final_output;
    double gpu_result;

    // 分配 GPU 全局内存
    cudaMalloc(&device_input, N * sizeof(double));
    cudaMalloc(&device_output, num_blocks * sizeof(double)); // 存储每个 block 的部分和
    cudaMalloc(&device_final_output, sizeof(double));        // 存储最终总和

    // 将数据从 CPU 拷贝到 GPU
    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ---------------- GPU 计时 ----------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 第一次核函数：每个 block 计算部分和
    reduce_v1<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output);

    // 第二次核函数：把上一步的结果再次归约，得到最终和
    reduce_v1<<<1, num_blocks>>>(device_output, device_final_output);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算 GPU 耗时（单位：毫秒）
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // 从 GPU 拷贝结果回主机
    cudaMemcpy(&gpu_result, device_final_output, sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "GPU time: " << gpu_time << " ms" << std::endl;
    std::cout << "GPU speedup: " << cpu_duration.count() / gpu_time << "x" << std::endl;

    // ---------------- 验证结果正确性 ----------------
    if (abs(cpu_result - gpu_result) < 1e-6)
    {
        std::cout << "Result verified: SUCCESS!" << std::endl;
    }
    else
    {
        std::cout << "Result verified: FAILED!" << std::endl;
    }

    // ---------------- 清理资源 ----------------
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_final_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
