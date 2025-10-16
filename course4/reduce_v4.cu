#include <chrono> // 用于 CPU 计时
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

// ======================================
// block_reduce: 块内归约函数（Block-level reduction）
// 使用 warp shuffle（线程间寄存器通信）实现高效归约
// ======================================
__inline__ __device__ float block_reduce(float val)
{
    const int tid = threadIdx.x; // 当前线程在线程块内的索引
    const int warpSize = 32;     // 每个 warp 包含 32 个线程
    int lane = tid % warpSize;   // 当前线程在所在 warp 内的 lane 索引
    int warpId = tid / warpSize; // 当前线程属于第几个 warp

    // 第一阶段：在每个 warp 内部使用 shuffle 指令进行归约
    // __shfl_down_sync 是一种 warp 内线程直接交换寄存器值的方式
    // offset 每次减半，相当于二叉树式归约
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // 声明共享内存，用于存储每个 warp 的部分和
    __shared__ float warp_sum[warpSize];

    // 每个 warp 的第一个线程（lane==0）将 warp 内的和写入共享内存
    if (lane == 0)
    {
        warp_sum[warpId] = val;
    }

    __syncthreads(); // 等待所有 warp 写入完共享内存

    // 第二阶段：使用第一个 warp 对 warp_sum 继续归约
    if (warpId == 0)
    {
        // 每个线程取一个 warp 的部分和
        // 超过 warp 数量的线程赋值为 0，防止越界
        val = (tid < blockDim.x / warpSize) ? warp_sum[tid] : 0.0f;

#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    // 返回块内最终的和（仅第一个线程得到完整结果）
    return val;
}

// ======================================
// reduce_v4: 每个 block 对输入数组的一部分进行归约
// 再次调用自身实现多级归约
// ======================================
__global__ void reduce_v4(const float* in, float* out, int n)
{
    float sum = 0.0f;

    // 每个线程负责处理多个元素（步长为 blockDim.x * gridDim.x）
    // 这样可以让线程充分利用全局内存带宽
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        sum += in[i];
    }

    // 调用 block_reduce 对块内线程的部分和进行归约
    sum = block_reduce(sum);

    // 每个块的第一个线程（threadIdx.x == 0）写出块内和
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

// ======================================
// reduce_cpu: CPU 版本的归约，用于验证 GPU 结果
// ======================================
float reduce_cpu(const std::vector<float>& data)
{
    return std::accumulate(data.begin(), data.end(), 0.0f);
}

// 每个 block 含 1024 个线程
const int BLOCK_SIZE = 1024;
// 总元素数量 1M
const int N = 1024 * 1024;

int main()
{
    // 计算所需 block 数量（向上取整）
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 准备输入数据：1.0, 2.0, 3.0, ..., N
    std::vector<float> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0f);

    // ---------------- CPU 归约 ----------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

    // ---------------- GPU 内存分配 ----------------
    float *device_input, *device_out, *device_final_out;
    float gpu_result;

    cudaMalloc(&device_input, N * sizeof(float));        // 输入数组
    cudaMalloc(&device_out, num_blocks * sizeof(float)); // 每个 block 输出一个部分和
    cudaMalloc(&device_final_out, sizeof(float));        // 存放最终结果

    cudaMemcpy(device_input, host_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 CUDA 计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------- GPU 归约阶段 ----------------
    cudaEventRecord(start);

    // 第一次调用：每个 block 对部分数据归约
    reduce_v4<<<num_blocks, BLOCK_SIZE>>>(device_input, device_out, N);

    // 第二次调用：用一个 block 对所有块的结果进行最终归约
    reduce_v4<<<1, num_blocks>>>(device_out, device_final_out, num_blocks);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_duration;
    cudaEventElapsedTime(&gpu_duration, start, stop);

    // 拷回 GPU 结果
    cudaMemcpy(&gpu_result, device_final_out, sizeof(float), cudaMemcpyDeviceToHost);

    // ---------------- 输出结果 ----------------
    std::cout << "GPU result: " << gpu_result << std::endl;
    std::cout << "GPU time: " << gpu_duration << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_duration.count() / gpu_duration << "x" << std::endl;

    // ---------------- 验证结果一致性 ----------------
    if ((abs(cpu_result - gpu_result) / max(abs(cpu_result), abs(gpu_result)) < 1e-3) ||
        abs(cpu_result - gpu_result) < 1e-3)
    {
        std::cout << "Result verified successfully!" << std::endl;
    }
    else
    {
        std::cout << "Result verification failed!" << std::endl;
    }

    // ---------------- 清理资源 ----------------
    cudaFree(device_input);
    cudaFree(device_out);
    cudaFree(device_final_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
