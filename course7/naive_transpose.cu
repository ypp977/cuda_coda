#include <cuda_runtime.h>
#include <iostream>

// ------------------------------------------------------------
// Kernel: naiveGmem
// 功能：将输入矩阵 in 做“转置”后写入 out。
//       in 的逻辑尺寸为 ny 行 × nx 列（row-major：in[iy * nx + ix]）
//       out 中存放转置后的矩阵，逻辑尺寸为 nx 行 × ny 列：out[ix * ny + iy]。
// 说明：
//   - 实现非常朴素（naive），只使用全局内存读写，没有任何共享内存优化；
//   - 访问模式并不完全共alesced，仅用于演示索引计算与转置关系。
// 参数：
//   out : 设备端输出矩阵指针，大小至少为 nx * ny
//   in  : 设备端输入矩阵指针，大小至少为 nx * ny
//   nx  : 列数（每行元素个数）
//   ny  : 行数
// ------------------------------------------------------------
__global__ void naiveGmem(float* out, float* in, int nx, int ny)
{
    // 计算当前线程负责的元素的二维坐标 (ix, iy)
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; // 列索引 [0, nx)
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; // 行索引 [0, ny)

    // 边界保护：只处理矩阵范围内的元素
    if (ix < (unsigned int)nx && iy < (unsigned int)ny)
    {
        // 输入矩阵按 row-major：行 iy，列 ix
        //   inIndex  = iy * nx + ix
        //
        // 输出矩阵存放转置结果：原列 ix → 新行 ix，原行 iy → 新列 iy
        //   outIndex = ix * ny + iy   （逻辑上是 [nx 行][ny 列] 的 row-major）
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// ------------------------------------------------------------
// 封装函数：配置 grid / block 并调用 naiveGmem kernel
//
// 参数：
//   d_out : 设备端输出矩阵指针
//   d_in  : 设备端输入矩阵指针
//   nx    : 列数
//   ny    : 行数
//
// 说明：
//   - 这里 blockSize 选用 (2, 2) 仅用于演示和便于手算索引，
//     实际应用中通常使用更大的 block（例如 16×16 或 32×8 等）。
// ------------------------------------------------------------
void call_naiveGmem(float* d_out, float* d_in, int nx, int ny)
{
    dim3 blockSize(2, 2); // 每个线程块 2×2 个线程

    // grid 维度按“向上取整”的方式覆盖整个矩阵
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    naiveGmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

int main()
{
    // -----------------------------
    // 1. 定义矩阵尺寸：nx 列、ny 行
    // -----------------------------
    int nx = 4;
    int ny = 4;
    size_t size = nx * ny * sizeof(float);

    // -----------------------------
    // 2. 在主机端分配输入 / 输出数组
    // -----------------------------
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    // -----------------------------
    // 3. 初始化输入矩阵 h_in
    //    使用简单的模式：h_in[i] = i % 11，方便观察转置前后结果
    //    逻辑布局为：ny 行 × nx 列，row-major
    // -----------------------------
    for (int i = 0; i < nx * ny; i++)
    {
        h_in[i] = float(int(i) % 11);
    }

    // -----------------------------
    // 4. 在设备端分配内存
    // -----------------------------
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // -----------------------------
    // 5. 将输入矩阵从主机拷贝到设备
    // -----------------------------
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // -----------------------------
    // 6. 调用 kernel 完成矩阵转置
    // -----------------------------
    call_naiveGmem(d_out, d_in, nx, ny);
    cudaDeviceSynchronize();

    // -----------------------------
    // 7. 将结果从设备拷贝回主机
    // -----------------------------
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // -----------------------------
    // 8. 打印原矩阵（按 ny 行 × nx 列输出）
    // -----------------------------
    std::cout << "Input matrix (row-major, " << ny << " x " << nx << "):\n";
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            std::cout << h_in[j * nx + i] << " ";
        }
        std::cout << "\n";
    }

    printf("---------------\n");

    // -----------------------------
    // 9. 打印输出数组当前的线性布局
    //
    // 说明：
    //   - kernel 将 in 的 (ny x nx) 转置为 out 的 (nx x ny)，
    //     out 的逻辑布局为“nx 行 × ny 列”，索引 out[row * ny + col]。
    //   - 这里依然按 j ∈ [0, ny), i ∈ [0, nx) 以 j*nx+i 的方式打印，
    //     只是为了方便观察 out 的线性存储内容，不是严格意义上的
    //     “按转置后二维形状”逐行打印。
    //   - 如果想以“矩阵形式”查看转置结果，应按 nx 行 × ny 列 去解读 out。
    // -----------------------------
    std::cout << "Output buffer (raw layout, printed as ny x nx for inspection):\n";
    for (int j = 0; j < ny; ++j)
    {
        for (int i = 0; i < nx; ++i)
        {
            std::cout << h_out[j * nx + i] << " ";
        }
        std::cout << "\n";
    }

    // -----------------------------
    // 10. 释放主机 / 设备内存
    // -----------------------------
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Matrix transposition completed." << std::endl;

    return 0;
}
