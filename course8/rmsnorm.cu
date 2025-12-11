// rmsnorm_cuda_test.cpp
// ---------------------------------------------
// 用于验证 / 对比 RMSNorm 的 CPU 与 CUDA 实现：
//   y = RMSNorm(x) * weight
//   其中：RMS(x) = sqrt( mean(x^2) + eps )，这里实现的是按 row（batch 维度）做 RMSNorm。
// ---------------------------------------------

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// ============================================================
// CPU 版本：按 row 维度做 RMSNorm（标量实现）
//   in    : [batch, size]
//   weight: [size]
//   out   : [batch, size]
//   eps   : 数值稳定项
// 公式：
//   rms = 1 / sqrt( (1/size) * sum_j x_j^2 + eps )
//   out_j = x_j * weight_j * rms
// ============================================================
void row_rmsnorm_f32_dim_cpu(float* in, float* weight, float* out, int batch, int size, float eps)
{
    for (int i = 0; i < batch; ++i)
    {
        // 当前 batch row 的起始指针
        float* in_ptr = in + i * size;
        float* out_ptr = out + i * size;

        // 1) 计算该 row 的平方和
        float sum = 0.0f;
        for (int j = 0; j < size; ++j)
        {
            float val = in_ptr[j];
            sum += val * val;
        }

        // 2) 计算缩放因子 scale = 1 / RMS
        float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

        // 3) 写出结果：先乘 weight，再乘 scale
        for (int j = 0; j < size; ++j)
        {
            float x = in_ptr[j] * weight[j];
            out_ptr[j] = x * rms;
        }
    }
}

// ============================================================
// block_reduce：对一个 block 内所有线程的 val 做求和规约
//
// 输入：val 为每个线程的局部求和
// 返回：
//   - 对于 warp_id==0 的所有线程：返回整块的最终和
//   - 其他 warp 的线程：返回 0.0f（避免误用）
//
// 核心步骤：
//   1) 先在每个 warp 内使用 __shfl_down_sync 做 warp 内规约。
//   2) warp 内 lane 0 把各 warp 的部分和写入共享内存 warpSums[warp_id]。
//   3) 再由第一个 warp 对 warpSums 做第二轮规约，得到 block 总和。
// 注意：
//   - 共享内存 warpSums[32] 假设一个 block 最多 32 个 warp（即 blockDim.x <= 32*32 = 1024）。
// ============================================================
__inline__ __device__ float block_reduce(float val)
{
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int lane = tid % warpSize;    // 线程在 warp 内的 lane id
    int warp_id = tid / warpSize; // 所属 warp 在 block 内的编号

    // ----------------------------
    // 1) warp 内规约：通过 shuffle 让同一 warp 内的线程交换寄存器值
    //    offset 依次为 16, 8, 4, 2, 1，实现 tree-reduction
    // ----------------------------
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // ----------------------------
    // 2) 每个 warp 的 lane 0 把自己的部分和写进共享内存
    // ----------------------------
    __shared__ float warpSums[32]; // 最多支持 32 个 warp 每块（1024 线程）
    if (lane == 0)
    {
        warpSums[warp_id] = val;
    }
    __syncthreads();

    // ----------------------------
    // 3) 只有第一个 warp 再对 warpSums 做一次规约，得到 block 总和
    // ----------------------------
    if (warp_id == 0)
    {
        // 计算参与第二轮规约的 warp 个数（即有多少有效的 warpSums）
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        // 超出 numWarps 的线程（tid >= numWarps）视为 0
        val = (tid < numWarps) ? warpSums[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    else
    {
        // 非第一个 warp 的线程，其返回值不再参与后续使用
        val = 0.0f;
    }
    return val;
}

// ============================================================
// Kernel 1：row_rmsnorm_f32_dim_simd
// 按 row 做 RMSNorm，使用 float4 向量化加载以提升内存带宽利用率。
// 每个 block 处理一个 row（一个 batch 行）。
//
// 要求：
//   - size >= 4 且最好是 4 的倍数（否则尾部用标量循环补齐）。
//   - in / wei / out 都按 float 对齐，reinterpret_cast<float4*> 的前提是
//     指针地址按 sizeof(float4) = 16 对齐，通常 cudaMalloc 能保证。
// ============================================================
__global__ void row_rmsnorm_f32_dim_simd(float* in, float* wei, float* out, int batch, int size,
                                         float eps)
{
    const int bid = blockIdx.x;  // 当前 block 对应的 batch 行号
    const int tid = threadIdx.x; // 当前线程在线程块内的索引

    // 一般配置 gridDim.x = batch，每个 block 负责一行
    if (bid >= batch)
    {
        return;
    }

    // 当前行在 in / out 中的起始指针
    float* block_in = in + bid * size;
    float* block_out = out + bid * size;

    // 以 4 元打包，前面尽量用 float4 做向量化访问，尾部不足 4 的用标量处理
    constexpr int pack_size = 4;
    const int pack_num = size / pack_size;     // 能整除的 float4 个数
    const int pack_off = pack_size * pack_num; // 向量化部分覆盖的元素个数

    // ----------------------------
    // 1) 计算当前 row 的平方和（sum = Σ x_i^2）
    //    先处理向量化部分，再处理尾部标量部分
    // ----------------------------
    float sum = 0.0f;
    float4* in_pack = reinterpret_cast<float4*>(block_in);

    // 1.1 向量化部分：每个线程 stride=blockDim.x 负责若干个 float4
    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    // 1.2 尾部标量部分：size 不是 4 的倍数时，剩余元素用标量处理
    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        sum += block_in[i] * block_in[i];
    }

    // ----------------------------
    // 2) 对该 row 上所有线程局部 sum 做 block 范围的规约
    //    得到本 row 的总平方和 sum = Σ_j x_j^2
    // ----------------------------
    __shared__ float shared_val; // 存储本 row 的最终 sum
    sum = block_reduce(sum);

    // block_reduce 的返回值对 warp0 有效，这里用 thread 0 把结果写进 shared_val
    if (threadIdx.x == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val; // 所有线程都拿到该 row 的总平方和

    // ----------------------------
    // 3) 计算缩放因子 scale = 1 / RMS
    // ----------------------------
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    // ----------------------------
    // 4) 写出结果：先向量化部分，再尾部标量部分
    // ----------------------------
    float4* wei_pack = reinterpret_cast<float4*>(wei);
    float4* out_pack = reinterpret_cast<float4*>(block_out);

    // 4.1 向量化部分：对每个 float4 做按元素的 scale * in * weight
    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 in_float4 = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i) =
            make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                        scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
    }

    // 4.2 尾部标量部分
    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        block_out[i] = wei[i] * block_in[i] * scale;
    }
}

// ============================================================
// Kernel 2：row_rmsnorm_f32_dim
// 按 row 做 RMSNorm，标量实现（不向量化，便于对比和验证正确性）
// 每个 block 仍然负责一个 row。
// ============================================================
__global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int batch, int size,
                                    float eps)
{
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;

    float* block_in = in + bid * size;
    float* block_out = out + bid * size;
    float sum = 0.0f;

    // 1) 先在该 row 内做部分平方和（每个线程处理若干个元素）
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float x = block_in[i];
        sum += x * x;
    }

    // 2) block_reduce 求出该 row 的总平方和
    __shared__ float shared_val;
    sum = block_reduce(sum);

    if (threadIdx.x == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // 3) 计算缩放因子
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    // 4) 写出结果：out = (in * weight) * scale
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float x = block_in[i] * wei[i];
        block_out[i] = x * scale;
    }
}

// ============================================================
// 误差统计：CPU 输出与 CUDA 输出之间的最大绝对误差
//   - 额外打印第一个误差 > 1.0f 的位置，用于 debug。
// ============================================================
float compute_max_error(const std::vector<float>& cpu_out, const std::vector<float>& cuda_out,
                        int n)
{
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        float err = std::abs(cpu_out[i] - cuda_out[i]);
        max_err = std::max(max_err, err);
        if (max_err > 1.f)
        {
            std::cout << "Error at index " << i << ": CPU = " << cpu_out[i]
                      << ", CUDA = " << cuda_out[i] << ", Error = " << err << "\n";
            break;
        }
    }
    return max_err;
}

// ----------------------------
// Main Function
// ----------------------------
int main()
{
    // 测试参数：
    //   batch: 行数（一次同时处理多少个向量）
    //   size : 每行向量长度
    const int batch = 16;
    const int size = 1024;
    const float eps = 1e-6f;
    const int total = batch * size;

    // ------------------------
    // 1) 分配 host 内存
    // ------------------------
    std::vector<float> h_input(total);
    std::vector<float> h_weight(size);
    std::vector<float> h_output_cpu(total);
    std::vector<float> h_output_cuda(total);

    // ------------------------
    // 2) 随机初始化输入数据和权重
    // ------------------------
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total; ++i)
    {
        h_input[i] = dis(gen);
    }
    for (int i = 0; i < size; ++i)
    {
        h_weight[i] = dis(gen);
    }

    // ------------------------
    // 3) CPU 版本计时与计算
    // ------------------------
    auto start = std::chrono::high_resolution_clock::now();
    row_rmsnorm_f32_dim_cpu(h_input.data(), h_weight.data(), h_output_cpu.data(), batch, size, eps);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU RMSNorm took " << duration.count() << " microseconds.\n";

    // ------------------------
    // 4) CUDA 端内存申请与数据拷贝
    // ------------------------
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------
    // 5) Kernel 启动配置
    //    这里为简单起见，一个 block 处理一个 row：
    //      grid.x  = batch
    //      block.x = 1024 （假定 size >= 1024 或通过 stride 访问覆盖所有元素）
    //    //    如果 size < blockDim.x，则仍然可用，只是很多线程空转。
    // ------------------------
    const int block_size = 1024;
    const int grid_size = batch; // One block per batch row
    dim3 grid(grid_size);
    dim3 block(block_size);

    // ------------------------
    // 6) 使用 CUDA event 计时 GPU 版本
    // ------------------------
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // 6.1 warm-up：多次运行 kernel，让 GPU 进入稳定状态
    int warpup = 10;
    for (int i = 0; i < warpup; i++)
    {
        // 可以切换成 row_rmsnorm_f32_dim_simd 做比较：
        // row_rmsnorm_f32_dim_simd<<<grid, block>>>(d_input, d_weight, d_output, batch, size, eps);
        row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch, size, eps);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != 0)
    {
        printf("cuda error:%d\n", err);
    }

    // 6.2 正式计时若干次执行，取平均
    cudaEventRecord(start_event);
    int test_iter = 10;
    for (int i = 0; i < test_iter; ++i)
    {
        // 如需测试 SIMD 版本可以替换下行：
        // row_rmsnorm_f32_dim_simd<<<grid, block>>>(d_input, d_weight, d_output, batch, size, eps);
        row_rmsnorm_f32_dim<<<grid, block>>>(d_input, d_weight, d_output, batch, size, eps);
    }
    cudaEventRecord(stop_event);

    // 等待 GPU 完成，并计算时间
    cudaEventSynchronize(stop_event);
    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start_event, stop_event); // 单位：毫秒

    // ------------------------
    // 7) 把 GPU 结果拷回 host
    // ------------------------
    cudaMemcpy(h_output_cuda.data(), d_output, total * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "CUDA RMSNorm took " << cuda_time * 1000 / test_iter << " microseconds.\n";

    // ------------------------
    // 8) 结果对比：CPU vs CUDA
    // ------------------------
    float max_error = compute_max_error(h_output_cpu, h_output_cuda, total);
    std::cout << "Max absolute error (CPU vs CUDA): " << max_error << "\n";

    // 可选：打印前若干个输出对比
    std::cout << "\nFirst 10 outputs (CPU vs CUDA):\n";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "CPU: " << h_output_cpu[i] << " | CUDA: " << h_output_cuda[i]
                  << " | Diff: " << std::abs(h_output_cpu[i] - h_output_cuda[i]) << "\n";
    }

    // ------------------------
    // 9) 资源清理
    // ------------------------
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
