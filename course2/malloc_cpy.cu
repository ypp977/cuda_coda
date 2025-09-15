#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    // 数组大小
    int N = 4;
    // 数组所占字节数
    size_t size = N * sizeof(int);

    // 在CPU端分配内存
    int *host_a, *host_b;
    host_a = (int*)malloc(size);
    host_b = (int*)malloc(size);

    // 初始化 host_a 数组
    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
    }

    // 在GPU端分配内存
    int* device_a;
    cudaMalloc(&device_a, size);

    // 将CPU上的 host_a 数组内容拷贝到GPU的  device_a
    cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

    // 将 GPU 上的 device_a 数组内容再拷贝回CPU的 host_b
    cudaMemcpy(host_b, device_a, size, cudaMemcpyDeviceToHost);

    // 打印 host_b 的内容，检查数据是否正确
    for (int i = 0; i < N; i++)
    {
        printf("host_b[%d] = %d\n", i, host_b[i]);
    }

    // 释放 GPU 显存
    cudaFree(device_a);

    // 释放 CPU 显存
    free(host_a);
    free(host_b);

    return 0;
}
