#include <cstdint> // uint8_t：无符号 8 bit 整型
#include <cuda_runtime.h>
#include <iostream>

// ------------------------------------------------------------
// Kernel: hist
// 功能：对输入数组 input 做直方图统计：
//       对每个输入值 v ∈ [0, 255]，累加到 hist[v]。
//
// 参数：
//   - input :  输入数据指针（uint8_t，有 0~255 共 256 种可能取值）
//   - hist  :  长度至少为 256 的整型数组，作为直方图计数（下标 0~255）
//   - n     :  input 数组的元素个数
//
// 实现要点：
//   1) 使用 grid-stride loop，使任意长度 n 都可以由任意规模 (grid, block)
//      覆盖，而不需要严格要求“一个线程负责一个元素”；
//   2) 使用 atomicAdd 保证多个线程同时更新同一 bin 时结果正确；
//   3) 对于 uint8_t，数值天然在 [0, 255]，无需额外范围检查。
// ------------------------------------------------------------
__global__ void hist(const uint8_t* input, int* hist, int n)
{
    // 该线程负责处理的“起始下标”
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // grid-stride 步长：一个“虚拟大线程”跨步访问的数据间隔
    int stride = gridDim.x * blockDim.x;

    // grid-stride loop：让每个物理线程处理若干个（间隔 stride 的）元素
    for (int idx = tid; idx < n; idx += stride)
    {
        uint8_t v = input[idx]; // 当前元素的取值，范围必在 [0, 255]

        // 多个线程可能同时命中同一个 v，因此必须使用原子加法
        atomicAdd(&hist[v], 1);
    }
}

int main()
{
    // 演示用的一个 3×3 小矩阵，总元素数量 size = 9
    int M = 3, N = 3;
    int size = M * N;

    // -----------------------------
    // 1. 在 Host 端准备输入数组
    // -----------------------------
    uint8_t* host_input = new uint8_t[size];

    // 构造一个简单的分布：
    // 1 出现 1 次，2 出现 2 次，3 出现 3 次，4 出现 2 次，5 出现 1 次
    host_input[0] = 1;
    host_input[1] = 2;
    host_input[2] = 3;
    host_input[3] = 2;
    host_input[4] = 3;
    host_input[5] = 4;
    host_input[6] = 3;
    host_input[7] = 4;
    host_input[8] = 5;

    // -----------------------------
    // 2. 在 Device 端分配内存
    // -----------------------------
    uint8_t* device_input = nullptr;
    int* device_hist = nullptr;

    cudaMalloc(&device_input, size * sizeof(uint8_t));
    cudaMalloc(&device_hist, 256 * sizeof(int));

    // hist 初始置零
    cudaMemset(device_hist, 0, 256 * sizeof(int));

    // -----------------------------
    // 3. 拷贝输入数据到 GPU
    // -----------------------------
    cudaMemcpy(device_input, host_input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // -----------------------------
    // 4. 启动 Kernel
    //
    //    这里只是演示：grid=2, block=2，总线程数=4。
    //    实际使用时，一般会选择更大的 blockDim (例如 128/256/512) 和
    //    根据 n 计算合适的 gridDim。
    // -----------------------------
    dim3 block_size(2);
    dim3 grid_size(2);
    hist<<<grid_size, block_size>>>(device_input, device_hist, size);
    cudaDeviceSynchronize();

    // -----------------------------
    // 5. 取回直方图结果到 Host，并打印部分 bin
    // -----------------------------
    int host_hist[256] = {0};
    cudaMemcpy(host_hist, device_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印 1~6 这几个 bin 的计数结果
    for (int i = 1; i <= 6; ++i)
    {
        printf("%d : %d\n", i, host_hist[i]);
    }

    // -----------------------------
    // 6. 释放资源
    // -----------------------------
    delete[] host_input;
    cudaFree(device_input);
    cudaFree(device_hist);

    return 0;
}
