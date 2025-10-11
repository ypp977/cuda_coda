#include <chrono>  // for timing
#include <cmath>   // for INFINITY
#include <cstdlib> // for malloc/free
#include <iostream>

// CPU implementation
void softmax_forward_cpu(float* out, const float* inp, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float max_val = -INFINITY;
        for (int j = 0; j < C; j++)
        {
            if (inp_row[j] > max_val)
            {
                max_val = inp_row[j];
            }
        }

        float sum = 0.f;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - max_val);
            sum += out_row[j];
        }

        float norm = 1.f / (float)sum;
        for (int j = 0; j < C; j++)
        {
            out_row[j] *= norm;
        }
    }
}

// CUDA kernel
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float max_val = -INFINITY;
        for (int j = 0; j < C; j++)
        {
            if (inp_row[j] > max_val)
            {
                max_val = inp_row[j];
            }
        }

        float sum = 0.f;
        for (int j = 0; j < C; j++)
        {
            out_row[j] = expf(inp_row[j] - max_val);
            sum += out_row[j];
        }

        for (int j = 0; j < C; j++)
        {
            out_row[j] /= (float)sum;
        }
    }
}

// 简单结果比较函数
bool compare_results(const float* cpu, const float* gpu, int N, int C, float epsilon = 1e-3f)
{
    for (int i = 0; i < N * C; ++i)
    {
        if (fabs(cpu[i] - gpu[i]) > epsilon)
        {
            printf("Mismatch at index: %d ,CPU=  %.3f ,GPU= %.3f ,diff= %.9f\n", i, cpu[i], gpu[i],
                   fabs(cpu[i] - gpu[i]));
            return false;
        }
    }
    return true;
}

__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    const float* x = inp + idx * C;

    float max_val = -INFINITY;
    for (int i = tid; i < C; i += block_size)
    {
        max_val = fmaxf(max_val, x[i]);
    }
    shared[tid] = max_val;
    __syncthreads();

    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared[0];

    for (int i = tid; i < C; i += block_size)
    {
        out[i + idx * C] = expf(x[i] - max_val);
    }

    x = out + idx * C;
    float sum = 0.f;
    for (int i = tid; i < C; i += block_size)
    {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    sum = shared[0];

    for (int i = tid; i < C; i += block_size)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

__device__ void warpReduceMax(float& val)
{
    for (int stride = warpSize / 2; stride > 0; stride /= 2)
    {
        val = fmaxf(__shfl_down_sync(0xffffffff, val, stride), val);
    }
}

__device__ void warpReduceSum(float& val)
{
    for (int stride = warpSize / 2; stride > 0; stride /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
}

__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C)
{
    // 行号
    int idx = blockIdx.x;
    // 表示线程在Block内的ID
    int tid = threadIdx.x;

    // 指向第 idx 行的输入数据
    const float* x = inp + idx * C;

    // 找到该行的最大值
    float max = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        max = fmaxf(x[i], max);
    }

    warpReduceMax(max);

    max = __shfl_sync(0xffffffff, max, 0);

    // 计算 exp(x - max)
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[i + idx * C] = expf(x[i] - max);
    }

    // 计算 exp 的和
    x = out + idx * C;

    float sum = 0.f;

    for (int i = tid; i < C; i += blockDim.x)
    {
        sum += x[i];
    }

    warpReduceSum(sum);

    sum = __shfl_sync(0xffffffff, sum, 0);

    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C)
{
    extern __shared__ float shared[];
    // 当前Block对应的索引
    int idx = blockIdx.x;
    // 当前线程在Block内的索引
    int tid = threadIdx.x;
    // 当前线程在当前Block内所属的warp索引
    int warpId = tid / 32;
    // 当前线程所在warp内的索引
    int laneId = tid % 32;
    // 每个Block内warp数量
    int warpsPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // 指向当前行的输入数据
    const float* x = inp + idx * C;

    // 分别计算每个线程局部最大值
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    // 归约计算每个warp内的最大值
    warpReduceMax(maxval);
    // 每个warp的最大值写入shared mem
    if (laneId == 0)
    {
        maxvals[warpId] = maxval;
    }

    __syncthreads();

    // warp 内归约得到整行最大值
    if (tid == 0)
    {
        float val = maxvals[0];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val = fmaxf(val, maxvals[i]);
        }
        maxvals[0] = val;
    }
    __syncthreads();

    float offset = maxvals[0];

    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }

    x = out + idx * C;
    float sumval = 0.f;
    for (int i = tid; i < C; i += blockDim.x)
    {
        sumval += x[i];
    }

    warpReduceSum(sumval);

    if (laneId == 0)
    {
        sumvals[warpId] = sumval;
    }
    __syncthreads();

    if (tid == 0)
    {
        float val = sumvals[0];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    float sum = sumvals[0];

    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

int main()
{
    int N = 1024;
    int C = 10240;
    size_t num_elements = (size_t)N * C;

    // 分配 CPU 内存
    float* inp = (float*)malloc(num_elements * sizeof(float));
    float* out_cpu = (float*)malloc(num_elements * sizeof(float));
    float* out_gpu = (float*)malloc(num_elements * sizeof(float));

    // 初始化输入
    for (int n = 0; n < N; ++n)
    {
        for (int j = 0; j < C; j++)
        {
            inp[n * C + j] = float(j % 1000) * 0.001f;
        }
    }

    // 分配 GPU 内存
    float *device_inp, *device_out;
    cudaMalloc((void**)&device_inp, num_elements * sizeof(float));
    cudaMalloc((void**)&device_out, num_elements * sizeof(float));
    cudaMemcpy(device_inp, inp, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // CPU 版本
    {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        softmax_forward_cpu(out_cpu, inp, N, C);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
        printf("CPU time: %.3f ms\n", cpu_time.count());
    }
    // GPU 版本 1 :一个线程处理一整行
    {
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        int block_size = 128;
        int block_num = (N + block_size - 1) / block_size;
        softmax_forward_kernel1<<<block_num, block_size>>>(device_out, device_inp, N, C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_1 = 0;
        cudaEventElapsedTime(&gpu_time_1, start, stop);

        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = compare_results(out_cpu, out_gpu, N, C);
        printf("[kernel1] Results match: %s,\n[kernel1] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_1);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU 版本 2 :使用shared memory 归约
    {
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        int block_size = 128;
        int block_num = N;
        size_t sharedmem_size = block_size * sizeof(float);
        softmax_forward_kernel2<<<block_num, block_size, sharedmem_size>>>(device_out, device_inp,
                                                                           N, C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_2 = 0;
        cudaEventElapsedTime(&gpu_time_2, start, stop);

        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = compare_results(out_cpu, out_gpu, N, C);
        printf("[kernel2] Results match: %s,\n[kernel2] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_2);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU 版本 3 : 使用 warp-level shuffle 优化归约
    {
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int block_size = 128;
        int block_num = N;

        cudaEventRecord(start);

        softmax_forward_kernel3<<<block_num, block_size>>>(device_out, device_inp, N, C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_3 = 0;
        cudaEventElapsedTime(&gpu_time_3, start, stop);

        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = compare_results(out_cpu, out_gpu, N, C);
        printf("[kernel3] Results match: %s,\n[kernel3] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU 版本4 ：使用warp + shared mem混合优化
    {
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int block_size = 128;
        int block_num = N;
        int warpsPerBlock = block_size / 32;
        size_t sharedmem_size = 2 * warpsPerBlock * sizeof(float);

        cudaEventRecord(start);

        softmax_forward_kernel4<<<block_num, block_size, sharedmem_size>>>(device_out, device_inp,
                                                                           N, C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time_4 = 0;
        cudaEventElapsedTime(&gpu_time_4, start, stop);

        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        bool ok = compare_results(out_gpu, out_cpu, N, C);
        printf("[kernel4] Results match: %s,\n[kernel4] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_4);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(device_inp);
    cudaFree(device_out);
    free(inp);
    free(out_cpu);
    free(out_gpu);

    return 0;
}
