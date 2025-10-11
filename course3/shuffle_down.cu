#include <cuda_runtime.h>

#include <iostream>

__global__ void test_shuf_down_sync(int* device_out, const int* device_in)
{
    // 从输入数组中读取当前线程对应位置的数据
    // threadIdx.x 是当前线程在线程块中的索引（0到31）
    int val = device_in[threadIdx.x];

    // 使用 __shfl_down_sync 函数进行warp级别的数据交换
    // 参数说明：
    // - 0xffffffff: 掩码，表示warp中的所有线程都参与操作
    // - val: 要交换的数据值
    // - 2: 向下移动的步长，当前线程将获取距离自己2个位置的下游线程的值
    // - 32: warp大小，指定参与操作的线程数量
    val = __shfl_down_sync(0xffffffff, val, 2, 32);

    // 将处理后的值写入输出数组对应位置
    device_out[threadIdx.x] = val;
}

int main()
{
    const int num_threads = 32;

    // 使用栈分配的数组，用于存放输入数据
    int host_in[num_threads];
    int host_out[num_threads];

    // 初始化输入数组
    for (int i = 0; i < num_threads; i++)
    {
        host_in[i] = i;
    }

    int *device_in, *device_out;

    // 在GPU中分配内存
    cudaMalloc(&device_in, num_threads * sizeof(int));
    cudaMalloc(&device_out, num_threads * sizeof(int));

    // 将输入数据拷贝到GPU
    cudaMemcpy(device_in, host_in, num_threads * sizeof(int), cudaMemcpyHostToDevice);

    // 调用核函数
    test_shuf_down_sync<<<1, num_threads>>>(device_out, device_in);

    // 将输出数据拷贝回CPU
    cudaMemcpy(host_out, device_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "results after __shfl_down_sync:\n";

    for (int i = 0; i < num_threads; i++)
    {
        std::cout << "host_out[" << i << "]" << host_out[i] << "\n";
    }

    // 释放内存
    cudaFree(device_in);
    cudaFree(device_out);

    return 0;
}
