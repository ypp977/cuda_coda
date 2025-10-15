#include <chrono> // 用于 CPU 计时
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>

const int BLOCK_SIZE = 1024; // 每个线程块的线程数
const int N = 1024 * 1024;   // 1M 个元素

#define FULL_MASK 0xffffffff // 用于 warp 内 shuffle 操作的掩码

// ---------------------------
// GPU warp 级别归约函数
// ---------------------------
// 该函数假定线程号 tid < 32，也就是只在一个 warp 内部运行。
// 输入：shared memory 中的缓存 cache。
// 输出：将该 warp 的求和结果写回 cache[0]。
__device__ void warpReduce(double* cache, unsigned int tid)
{
    // 每个线程先把自己负责的元素与 tid+32 的元素相加
    // （假设 blockDim.x >= 64）
    double val = cache[tid] + cache[tid + 32];

    // 利用 warp shuffle 指令在同一个 warp 内进行树形归约
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        // 每次将当前线程的值与距离 offset 的线程的值相加
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }

    // 最终只有线程 tid == 0 会把本 warp 的求和结果写入共享内存的第 0 个位置
    if (tid == 0)
    {
        cache[0] = val;
    }
}

// ---------------------------
// CPU 端归约函数，用于对比 GPU 结果是否正确
// ---------------------------
double reduce_cpu(const std::vector<double>& data)
{
    // std::accumulate 用于把所有元素累加
    return std::accumulate(data.begin(), data.end(), 0.0);
}

// ---------------------------
// GPU 端核函数：块内归约
// ---------------------------
// 输入：data_input 是输入数组
// 输出：data_out 保存每个 block 的部分和
__global__ void reduce_v3(double* data_input, double* data_out)
{
    // 分配块内共享内存，用于存放本 block 的数据
    __shared__ double shared_data[BLOCK_SIZE];

    int tid = threadIdx.x;                     // 当前线程在块内的索引
    int i = blockIdx.x * blockDim.x * 2 + tid; // 每个线程处理两个元素

    // 每个线程先把两个元素的值加起来，存入共享内存
    shared_data[tid] = data_input[i] + data_input[i + blockDim.x];
    __syncthreads(); // 保证所有线程写完

    // 块内规约（树形加法）：每次循环折半参与的线程数
    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            shared_data[tid] += shared_data[tid + s]; // 累加对应偏移的数据
        }
        __syncthreads(); // 每轮都要同步，防止数据覆盖
    }

    // 当剩下 32 个线程时（一个 warp），调用 warpReduce 继续规约
    if (tid < 32)
    {
        warpReduce(shared_data, tid);
    }

    // 最终由线程 0 把该 block 的部分和写入输出数组
    if (tid == 0)
    {
        data_out[blockIdx.x] = shared_data[0];
    }
}

// ---------------------------
// 主函数
// ---------------------------
int main()
{
    // 计算需要多少个 block，每个 block 处理两个 BLOCK_SIZE 的元素
    int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) / 2;

    // 初始化主机端数据：填充为 [1.0, 2.0, 3.0, ...]
    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 1.0);

    // ---------------------------
    // CPU 求和并计时
    // ---------------------------
    auto cpu_start = std::chrono::high_resolution_clock::now();
    double cpu_result = reduce_cpu(host_data);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;

    std::cout << "CPU result :" << cpu_result << std::endl;
    std::cout << "CPU time :" << cpu_duration.count() << "ms" << std::endl;

    // ---------------------------
    // 分配 GPU 内存
    // ---------------------------
    double *device_input, *device_output, *device_final_output;
    double gpu_result;

    cudaMalloc(&device_input, N * sizeof(double));
    cudaMalloc(&device_output, num_blocks * sizeof(double));
    cudaMalloc(&device_final_output, sizeof(double));

    // 把输入数据拷贝到 GPU
    cudaMemcpy(device_input, host_data.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    // ---------------------------
    // 创建 CUDA 事件，用于计时 GPU 执行时间
    // ---------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); // 开始计时

    // 第一次 kernel 调用：对原始数据分块求部分和
    reduce_v3<<<num_blocks, BLOCK_SIZE>>>(device_input, device_output);

    // 第二次 kernel 调用：对第一次结果再进行一次归约得到最终结果
    reduce_v3<<<1, num_blocks>>>(device_output, device_final_output);

    cudaEventRecord(stop);      // 停止计时
    cudaEventSynchronize(stop); // 等待 GPU 完成

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop); // 计算 GPU 执行时间

    // 从 GPU 拷回最终结果
    cudaMemcpy(&gpu_result, device_final_output, sizeof(double), cudaMemcpyDeviceToHost);

    // ---------------------------
    // 校验 GPU 结果是否正确
    // ---------------------------
    if (abs(cpu_result - gpu_result) < 1e-6)
    {
        std::cout << "GPU result is correct" << std::endl;
    }
    else
    {
        std::cout << "GPU result is wrong" << std::endl;
    }

    // 输出 GPU 结果和加速比
    std::cout << "GPU result :" << gpu_result << std::endl;
    std::cout << "GPU time :" << gpu_time << "ms" << std::endl;
    std::cout << "GPU speedup :" << cpu_duration.count() / gpu_time << "x" << std::endl;

    // ---------------------------
    // 清理资源
    // ---------------------------
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_final_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
