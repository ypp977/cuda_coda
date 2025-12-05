#include <cstdint> // int8_t
#include <cuda_runtime.h>
#include <iostream>

// ------------------------------------------------------------
// Kernel: hist
// 功能：对输入数组 input 做直方图统计，将每个取值 in 出现的次数累加到 hist[in]。
//   - input:  输入数据（8 bit，有符号）
//   - hist:   长度至少为 256 的整型数组，作为直方图计数（下标 0~255）
//   - n:      input 数组的元素个数
//
// 实现要点：
//   1) 使用 grid-stride loop，让任意规模的输入都可以被覆盖。
//   2) 使用 atomicAdd 保证多个线程同时更新同一个 bin 时的正确性。
// ------------------------------------------------------------
__global__ void hist(int8_t* input, int* hist, int n)
{
    // 全局线程索引：当前线程负责的第一个元素下标
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // grid-stride loop：每次跨过整个网格覆盖的元素数量
    // 步长 = blockDim.x * gridDim.x（所有线程数）
    for (int idx = i; idx < n; idx += gridDim.x * blockDim.x)
    {
        // 读取当前线程负责的输入元素
        int8_t in = input[idx];

        // 防御性检查：
        //   - int8_t 的范围是 [-128, 127]，这里额外判断 in 在 [0, 255) 区间
        //   - 实际上 int8_t 不可能 >= 128，这里只是为了示意“合法 bin 范围是 0~255”
        if (in >= 0 && in < 256)
        {
            // 由于多个线程可能同时对同一个 bin 做加法，
            // 这里使用原子操作 atomicAdd 避免竞争条件。
            // in 是 int8_t，这里会被提升为 int 用作下标。
            atomicAdd(&hist[in], 1);
        }
    }
}

int main()
{
    // ------------------------------
    // 1. 构造一个 3x3 的小矩阵作为输入数据
    // ------------------------------
    int M = 3;
    int N = 3;
    int size = M * N; // 输入元素总数 = 9

    // 在 host 端分配输入数组
    int8_t* input = new int8_t[size];

    // 手动填充 3x3 数据：
    // [ 1 2 3
    //   2 3 4
    //   3 4 5 ]
    input[0] = 1;
    input[1] = 2;
    input[2] = 3;

    input[3] = 2;
    input[4] = 3;
    input[5] = 4;

    input[6] = 3;
    input[7] = 4;
    input[8] = 5;

    // ------------------------------
    // 2. 在 device 端分配内存
    // ------------------------------
    int8_t* d_input;
    int* d_hist;
    // 输入数据缓冲区
    cudaMalloc(&d_input, size * sizeof(int8_t));
    // 直方图缓冲区：256 个 bin，对应 0~255
    cudaMalloc(&d_hist, 256 * sizeof(int));

    // 将直方图缓冲区初始化为 0
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // ------------------------------
    // 3. 拷贝输入数据到 GPU
    // ------------------------------
    cudaMemcpy(d_input, input, sizeof(int8_t) * size, cudaMemcpyHostToDevice);

    // ------------------------------
    // 4. 启动 Kernel
    //    这里简单地用 2 blocks × 2 threads，共 4 个线程。
    //    通过 grid-stride loop，4 个线程会把 9 个元素遍历完。
    // ------------------------------
    dim3 block_size(2);
    dim3 grid_size(2);
    hist<<<grid_size, block_size>>>(d_input, d_hist, size);
    cudaDeviceSynchronize();

    // 检查 kernel 是否有运行时错误（如非法访问等）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cuda error: %d (%s)\n", err, cudaGetErrorString(err));
    }

    // ------------------------------
    // 5. 将直方图从 device 拷贝回 host
    // ------------------------------
    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // ------------------------------
    // 6. 打印直方图中 1~6 这几个 bin 的计数
    //    理论结果：
    //      1 出现 1 次
    //      2 出现 2 次
    //      3 出现 3 次
    //      4 出现 2 次
    //      5 出现 1 次
    //      6 出现 0 次
    // ------------------------------
    for (int i = 1;i <= 6; ++i)
    {
        printf("%d : %d\n", i, h_hist[i]);
    }

    // ------------------------------
    // 7. 释放 host 内存（device 内存在本例中就不再演示释放了）
    // ------------------------------
    delete[] input;

    return 0;
}
