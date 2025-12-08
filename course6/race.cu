#include <cuda_runtime.h>
#include <stdio.h>

/*
----------------------------------------------------------
Kernel: race_condition_kernel
作用：
  - 所有线程对同一个全局整型变量执行“读-改-写”：
        temp = *data;
        temp = temp + 1;
        *data = temp;
  - 故意不使用原子操作，制造典型的 data race 场景，用于演示竞态条件。

关键点：
  - 读-改-写是一个“复合操作”，在硬件层面会拆成多条指令：
      1) 从 global memory 读出 *data 到寄存器 temp
      2) temp + 1
      3) 将 temp 写回 *data
    这些步骤对不同线程是交叉执行的，没有互斥保护。
  - 多个线程可能读到相同的旧值，然后各自 +1 再写回，
    导致“丢失更新”（lost update），最终结果显著小于线程总数。
----------------------------------------------------------
*/
__global__ void race_condition_kernel(int* data)
{
    // 所有线程都对同一个地址 data 执行非原子的“读-改-写”
    int temp = *data; // 1) 从全局内存读取当前值
    temp = temp + 1;  // 2) 在寄存器中加 1
    *data = temp;     // 3) 将结果写回全局内存（可能覆盖其他线程的更新）
}

int main()
{
    int* d_data = nullptr;
    int h_data = 0;

    // --------------------------------------------------
    // 1. 在设备端为一个整型计数器分配内存，并初始化为 0
    // --------------------------------------------------
    cudaMalloc(&d_data, sizeof(int));
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    // --------------------------------------------------
    // 2. 启动 kernel
    //
    //    配置：
    //      gridDim  = 1024 blocks
    //      blockDim = 256 threads per block
    //
    //    总线程数 = 1024 * 256 = 262144
    //    如果没有任何竞态条件，理论上最终结果应该是 262144。
    // --------------------------------------------------
    dim3 grid(1024);
    dim3 block(256);
    race_condition_kernel<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    // --------------------------------------------------
    // 3. 从设备端拷贝结果回主机
    // --------------------------------------------------
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    // --------------------------------------------------
    // 4. 打印结果
    //
    //    - 期望值（无竞态、使用原子操作/锁时）：1024 * 256
    //    - 实际值：通常远小于期望值，而且多次运行结果不稳定，
    //      正是 data race 导致“丢失更新”的表现。
    //
    //    如果将 kernel 中的三行改为：
    //        atomicAdd(data, 1);
    //      则结果会稳定接近 1024 * 256，说明原子操作消除了竞态。
    // --------------------------------------------------
    int expected = 1024 * 256;
    printf("Final value: %d (expected %d if there were no race condition)\n", h_data, expected);

    // --------------------------------------------------
    // 5. 资源释放
    // --------------------------------------------------
    cudaFree(d_data);

    return 0;
}
