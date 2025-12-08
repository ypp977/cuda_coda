#include <cuda_runtime.h>
#include <stdio.h>

/*
----------------------------------------------------------
Kernel: race_condition_kernel
作用：所有线程对同一个全局变量做加 1 操作。

说明：
  - 这里使用 atomicAdd，对同一地址的并发写入是“串行化”的，
    因此最终结果是确定的，不会发生数据竞争（race condition）。
  - 如果把 atomicAdd 改成普通的 *data += 1，则会产生典型的
    读-改-写竞态，最终结果 < 线程总数。
----------------------------------------------------------
*/
__global__ void race_condition_kernel(int* data)
{
    // 所有线程并发地对同一个地址执行“原子加 1”
    atomicAdd(data, 1);
}

int main()
{
    int* d_data = nullptr;
    int h_data = 0;

    // -----------------------------
    // 1. 在设备端分配一个 int，用作全局计数器
    // -----------------------------
    cudaMalloc(&d_data, sizeof(int));

    // -----------------------------
    // 2. 初始化计数器为 0
    // -----------------------------
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    // -----------------------------
    // 3. 启动 kernel
    //
    //    网格配置：
    //      gridDim  = 1024 blocks
    //      blockDim = 256 threads per block
    //
    //    线程总数 = 1024 * 256 = 262144
    //    每个线程执行 atomicAdd(data, 1)，
    //    理论期望最终结果 = 262144。
    // -----------------------------
    dim3 grid(1024);
    dim3 block(256);
    race_condition_kernel<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    // -----------------------------
    // 4. 将结果拷贝回 Host
    // -----------------------------
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // -----------------------------
    // 5. 打印结果
    //    由于使用了原子操作，h_data 应该等于 1024 * 256。
    //    如果改成非原子操作（例如 *data += 1），
    //    则会看到明显小于期望值的结果，体现竞态问题。
    // -----------------------------
    int expected = 1024 * 256;
    printf("Final value: %d (expected %d with atomicAdd)\n", h_data, expected);

    // -----------------------------
    // 6. 资源清理
    // -----------------------------
    cudaFree(d_data);

    return 0;
}
