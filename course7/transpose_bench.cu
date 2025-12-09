#include <cuda_runtime.h>
#include <iostream>

// ============================================================
// Kernel 0: naiveGmem
// 仅使用全局内存的朴素矩阵转置实现
//
// in  视为 ny 行 × nx 列（row-major）：in[iy * nx + ix]
// out 视为 nx 行 × ny 列（row-major）：out[row * ny + col]
//      row = ix，col = iy
//
// 注意：这里没有做共享内存优化，目的是提供带宽下限的基准。
// ============================================================
__global__ void naiveGmem(float* out, float* in, int nx, int ny)
{
    // 原矩阵中的二维坐标 (ix, iy)
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; // 列索引
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; // 行索引

    // 边界保护：只在矩阵范围内读写
    if (ix < (unsigned int)nx && iy < (unsigned int)ny)
    {
        // 输入矩阵：ny × nx
        //   输入索引 = iy * nx + ix
        // 输出矩阵：nx × ny（in 的转置）
        //   输出索引 = ix * ny + iy
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

// ============================================================
// Block 尺寸配置（共享内存版本共用）
// BDIMX：block 内 X 方向线程数
// BDIMY：block 内 Y 方向线程数
// tile 尺寸为 BDIMY × BDIMX
// ============================================================
#define BDIMX 32
#define BDIMY 16

// ============================================================
// Kernel 1: transposeSmem
// 使用二维共享内存 tile 做转置，但不做 bank conflict padding
//
// 思路：
//   1) 原矩阵按 BDIMY×BDIMX 的 tile 从全局内存读入 shared memory。
//      读入时访问 in[iy * nx + ix]，保证读路径尽量 coalesced。
//   2) 再按“转置后”的线程映射从共享内存读出 tile[icol][irow]，
//      写入 out 的转置位置。
//   3) 由于 shared memory 以 [row][col] 形式声明，当按列访问时会有
//      严重的 bank conflict，本 kernel 用于展示“只有 shared mem，
//      但没做 padding 的情况”。
// 说明：
//   - 代码中用于输出的边界判断 if (ix < nx && iy < ny) 只在 nx == ny 时严格正确，
//     用于 4096×4096 这类方阵是没问题的；要支持长宽不等需进一步修正边界条件。
// ============================================================
__global__ void transposeSmem(float* out, float* in, const int nx, const int ny)
{
    // 共享内存 tile，形状为 BDIMY × BDIMX
    __shared__ float tile[BDIMY][BDIMX];

    // ---------- 原矩阵坐标 ----------
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; // 原矩阵列
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; // 原矩阵行

    // 原矩阵线性下标
    unsigned int ti = iy * nx + ix;

    // ---------- 在线程块内部重新编号，方便构造“转置块内坐标” ----------
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;

    // irow / icol 是“转置后块内坐标”，相当于把一个 BDIMY×BDIMX 的 tile
    // 按行展平再重新映射到 BDIMX×BDIMY 的转置布局。
    unsigned int irow = bidx / blockDim.y; // [0, BDIMX)
    unsigned int icol = bidx % blockDim.y; // [0, BDIMY)

    // ---------- 转置矩阵坐标 ----------
    // 交换 blockIdx.x / blockIdx.y，并在块内使用 (icol, irow) 作为坐标
    ix = blockIdx.y * blockDim.y + icol; // 转置后列索引（原来的行方向）
    iy = blockIdx.x * blockDim.x + irow; // 转置后行索引（原来的列方向）

    // 转置矩阵线性下标
    // out 逻辑尺寸为：nx 行 × ny 列
    unsigned int to = iy * ny + ix;

    // 注：严格来说这里应该用 if (ix < ny && iy < nx) 才能支持非方阵。
    if (ix < (unsigned int)nx && iy < (unsigned int)ny)
    {
        // 1) 先将原矩阵中的当前元素写入共享内存
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();

        // 2) 再按“转置后的块内坐标”从共享内存读出
        out[to] = tile[icol][irow];
    }
}

// ============================================================
// Kernel 2: transposeSmemUnpad
// 在共享内存 tile 的 X 方向增加 +1 列 padding，用于缓解 bank conflict
//
// 差异点：
//   - tile 声明为 [BDIMY][BDIMX + pad]，pad = 1
//   - 读写逻辑与 transposeSmem 相同，只是共享内存布局不同。
//   - 访问 [row][col] 对应到不同 bank 时，padding 打乱了最坏冲突模式。
// ============================================================
__global__ void transposeSmemUnpad(float* out, float* in, const int nx, const int ny)
{
    const int pad = 1;
    __shared__ float tile[BDIMY][BDIMX + pad];

    // ---------- 原矩阵坐标 ----------
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // 原矩阵线性下标
    unsigned int ti = iy * nx + ix;

    // ---------- 块内索引重映射 ----------
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // ---------- 转置矩阵坐标 ----------
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    unsigned int to = iy * ny + ix;

    // 同样，这里的边界条件只在 nx == ny 时严格对。
    if (ix < (unsigned int)nx && iy < (unsigned int)ny)
    {
        // 共享内存写入使用 [threadIdx.y][threadIdx.x]，受益于 padding
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();

        // 从“转置后的”共享内存坐标读出
        out[to] = tile[icol][irow];
    }
}

// ============================================================
// Kernel 3: transposeSmemUnrollPad
// 使用一维共享内存 + padding + X 方向 2 倍 unroll 的版本
//
// 设计要点：
//   1) 每个 block 在 X 方向一次处理 2*BDIMX 列（ix = 2*BDIMX*bx + tx）。
//      这样一个线程可以一次从全局内存读两个元素（in[ti], in[ti + BDIMX]）。
//   2) 共享内存 tile 以一维数组形式声明：
//        tile[BDIMY * (BDIMX*2 + IPAD)]
//      逻辑上表示 BDIMY 行、(2*BDIMX+IPAD) 列的二维 tile。
//   3) 读：每个线程向 tile 的一行写入两个元素（未转置，行主序），
//      确保全局内存读是 coalesced。
//   4) 写：通过 col_idx 映射，从共享内存按“列”方向读出，实现 tile 内转置，
//      再写回 out 的两个位置（to、to + ny*BDIMX）。
//   5) 仍然是方阵假设下使用（nx == ny），主要用于比较带宽上限。
// ============================================================
__global__ void transposeSmemUnrollPad(float* out, float* in, int nx, int ny)
{
    const int IPAD = 1;
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // ---------- 原矩阵坐标（一次处理 2*BDIMX 宽度） ----------
    unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // 原矩阵线性下标
    unsigned int ti = iy * nx + ix;

    // 块内线性线程编号
    unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int irow = bidx / blockDim.y; // [0, 2*BDIMX)
    unsigned int icol = bidx % blockDim.y; // [0, BDIMY)

    // ---------- 转置后矩阵坐标 ----------
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;     // 转置后列
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow; // 转置后行

    unsigned int to = iy2 * ny + ix2;

    // 这里要求：(ix + BDIMX) < nx，确保一次 unroll 访问的两列都在范围内
    if ((ix + blockDim.x) < (unsigned int)nx && iy < (unsigned int)ny)
    {
        // ---------- 写 tile：按“行主序”写入两列 ----------
        // row_idx 对应共享内存中的行起始下标
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;

        // 当前线程负责一行中的两个元素：tile[row_idx] 和 tile[row_idx + BDIMX]
        tile[row_idx] = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        __syncthreads();

        // ---------- 读 tile：按“列主序”读出，实现 tile 内转置 ----------
        // icol 是“转置后的行”，irow 是“转置后的列”，因此以 icol 作为行索引、
        // 以 irow 作为列索引构造 col_idx。
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;

        // 写回 out 的两个位置，对应原来 2*BDIMX 宽度的两列
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}

// ============================================================
// 封装函数：分别配置 grid/block 并调用各版本 kernel
// ============================================================

// 朴素全局内存版本
void call_naiveGmem(float* d_out, float* d_in, int nx, int ny)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((nx + blockSize.x - 1) / blockSize.x, (ny + blockSize.y - 1) / blockSize.y);

    naiveGmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

// 共享内存（无 padding）版本
void call_transposeSmem(float* d_out, float* d_in, const int nx, const int ny)
{
    dim3 blockSize(BDIMX, BDIMY);
    dim3 gridSize((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

    transposeSmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

// 共享内存 + padding 版本
void call_transposeSmemUnpad(float* d_out, float* d_in, const int nx, const int ny)
{
    dim3 blockSize(BDIMX, BDIMY);
    dim3 gridSize((nx + BDIMX - 1) / BDIMX, (ny + BDIMY - 1) / BDIMY);

    transposeSmemUnpad<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

// 共享内存 + padding + X 方向 2×unroll 版本
void call_transposeSmemUnrollUnpad(float* d_out, float* d_in, const int nx, const int ny)
{
    dim3 blockSize(BDIMX, BDIMY);

    // 每个 block 在 X 方向处理 2*BDIMX 列，因此 grid.x 约为 nx / (2*BDIMX)
    auto grid_x_full = (nx + BDIMX - 1) / BDIMX;
    dim3 gridSize(int(grid_x_full / 2), (ny + BDIMY - 1) / BDIMY);

    transposeSmemUnrollPad<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

// ============================================================
// 下列 wrapper 负责：
//   - 分配/初始化 host & device 内存
//   - 预热 kernel
//   - 使用 CUDA event 做简单时间统计
//   - 不做结果校验，仅用于性能对比
//   - 默认以 4096×4096 方阵测试（满足前面若干实现的方阵假设）
// ============================================================

void naiveGmemWrapper()
{
    int nx = 4096;
    int ny = 4096;
    size_t size = size_t(nx) * ny * sizeof(float);

    // 1) 分配并初始化 host 端数据
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    for (int i = 0; i < nx * ny; i++)
        h_in[i] = float(i % 11); // 简单模式，便于后续调试可视化

    // 2) 分配 device 端内存并拷贝输入
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 3) 创建 CUDA event，用于测量 kernel 执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warm_up_iter = 5;
    // 4) 预热，避免首次启动 overhead 干扰
    for (int i = 0; i < warm_up_iter; ++i)
        call_naiveGmem(d_out, d_in, nx, ny);

    int bench_iter = 5;

    // 5) 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < bench_iter; ++i)
        call_naiveGmem(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 6) 检查 kernel 是否有运行时错误（例如非法访问）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 7) 统计平均运行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive gmem transpose: " << milliseconds / float(bench_iter) << " ms\n";

    // 8) 拷回结果（本例不做数值校验，仅占位）
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 9) 资源释放
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Naive gmem matrix transposition completed.\n";
}

void naiveSmemWrapper()
{
    int nx = 4096;
    int ny = 4096;
    size_t size = size_t(nx) * ny * sizeof(float);

    // 1) host 端分配与初始化
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    for (int i = 0; i < nx * ny; i++)
        h_in[i] = float(i % 11);

    // 2) device 端分配与数据拷贝
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 3) 创建计时用 event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warm_up_iter = 5;
    // 4) 预热共享内存版本 kernel
    for (int i = 0; i < warm_up_iter; ++i)
        call_transposeSmem(d_out, d_in, nx, ny);

    int bench_iter = 5;

    // 5) 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < bench_iter; ++i)
        call_transposeSmem(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 6) 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 7) 输出平均耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Smem transpose (no pad): " << milliseconds / float(bench_iter) << " ms\n";

    // 8) 拷贝结果回 host（不做数值校验）
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 9) 资源释放
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Smem matrix transposition completed.\n";
}

void naiveSmemWrapperUnpad()
{
    int nx = 4096;
    int ny = 4096;
    size_t size = size_t(nx) * ny * sizeof(float);

    // 1) host 端分配与初始化
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    for (int i = 0; i < nx * ny; i++)
        h_in[i] = float(i % 11);

    // 2) device 端分配与数据拷贝
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 3) 创建计时用 event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warm_up_iter = 5;
    // 4) 预热“共享内存 + padding”版本 kernel
    for (int i = 0; i < warm_up_iter; ++i)
        call_transposeSmemUnpad(d_out, d_in, nx, ny);

    int bench_iter = 5;

    // 5) 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < bench_iter; ++i)
        call_transposeSmemUnpad(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 6) 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 7) 输出平均耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Smem transpose (pad): " << milliseconds / float(bench_iter) << " ms\n";

    // 8) 拷贝结果回 host（不做数值校验）
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 9) 资源释放
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Smem (pad) matrix transposition completed.\n";
}

void naiveSmemWrapperUnrollUnpad()
{
    int nx = 4096;
    int ny = 4096;
    size_t size = size_t(nx) * ny * sizeof(float);

    // 1) host 端分配与初始化
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);

    for (int i = 0; i < nx * ny; i++)
        h_in[i] = float(i % 11);

    // 2) device 端分配与数据拷贝
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 3) 创建计时用 event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int warm_up_iter = 5;
    // 4) 预热“共享内存 + padding + 2x unroll”版本
    for (int i = 0; i < warm_up_iter; ++i)
        call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);

    int bench_iter = 5;

    // 5) 正式计时
    cudaEventRecord(start);
    for (int i = 0; i < bench_iter; ++i)
        call_transposeSmemUnrollUnpad(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 6) 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 7) 输出平均耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Smem transpose (2x unroll + pad): " << milliseconds / float(bench_iter)
              << " ms\n";

    // 8) 拷贝结果回 host（不做数值校验）
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 9) 资源释放
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    std::cout << "Smem (2x unroll + pad) matrix transposition completed.\n";
}

int main()
{
    // 根据需要测试不同版本：
    // naiveGmemWrapper();
    // naiveSmemWrapper();
    // naiveSmemWrapperUnpad();
    naiveSmemWrapperUnrollUnpad();
    return 0;
}
