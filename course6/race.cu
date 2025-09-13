#include <stdio.h>
#include <cuda_runtime.h>

__global__ void race_condition_kernel(int *data)
{
    // 所有线程都尝试写入同一个内存位置
    int temp = *data;
    temp = temp + 1;
    *data = temp;
}

int main()
{
    int *d_data;
    int h_data = 0;

    // 分配设备内存
    cudaMalloc(&d_data, sizeof(int));

    // 初始化设备数据
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    // 使用 1024 个线程块，每个块 256 个线程
    race_condition_kernel<<<1024, 256>>>(d_data);
    cudaDeviceSynchronize();

    // 将结果拷贝回主机
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Final value: %d (expected %d if no race condition)\n",
           h_data, 1024 * 256);

    // 清理
    cudaFree(d_data);

    return 0;
}
