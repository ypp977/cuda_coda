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
    for (int i = 0; i < N; ++i)
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

__device__ float warpReduceSum(float val)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
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

    max = __shfl_sync(0xfffffff, max, 0);

    // 计算

    float
}

__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C)
{
    // out is (N, C) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of C elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations warpReduceMax/warpReduceSum are used for
    // intra-warp reductions shared memory is used for inter-warp reduction
    extern __shared__ float shared[];
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32; // warp index within a block
    int laneId = threadIdx.x % 32; // thread index within a warp

    // the number of warps per block. recall that blockDim.x is block_size
    int warpsPerBlock = blockDim.x / 32;

    // shared[] must be allocated to have 2 * warpsPerBlock elements
    // first half for max values, the second half for sum values
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // one row of inp, i.e. inp[idx, :] of shape (C,)
    const float* x = inp + idx * C;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        maxval = fmaxf(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = warpReduceMax(maxval);

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0)
        maxvals[warpId] = maxval;
    __syncthreads();

    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0)
    {
        float val = maxvals[tid];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            val = fmaxf(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    __syncthreads();
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = expf(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + idx * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += blockDim.x)
    {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = warpReduceSum(sumval);

    // write sumval to shared memory
    if (laneId == 0)
        sumvals[warpId] = sumval;
    __syncthreads();

    // inter-thread reduction of sum
    if (tid == 0)
    {
        float val = sumvals[tid];
        for (int i = 1; i < warpsPerBlock; ++i)
        {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    __syncthreads();
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (int i = tid; i < C; i += blockDim.x)
    {
        out[idx * C + i] = x[i] / sum;
    }
}

// int main()
// {
//     // Example: batch size N=32, classes C=4096
//     int N = 32;
//     int C = 4096;

//     size_t num_elements = N * C;
//     float* inp = (float*)malloc(num_elements * sizeof(float));
//     float* out_cpu = (float*)malloc(num_elements * sizeof(float));
//     float* out_gpu = (float*)malloc(num_elements * sizeof(float));

//     // Initialize input with sample data
//     for (int n = 0; n < N; ++n)
//     {
//         for (int c = 0; c < C; ++c)
//         {
//             inp[n * C + c] = float(c);
//         }
//     }

//     // Run CPU version and measure time
//     auto start_cpu = std::chrono::high_resolution_clock::now();
//     softmax_forward_cpu(out_cpu, inp, N, C);
//     auto end_cpu = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

//     // Run GPU version and measure time using CUDA events
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     float *d_out, *d_inp;
//     cudaMalloc((void**)&d_out, N * C * sizeof(float));
//     cudaMalloc((void**)&d_inp, N * C * sizeof(float));
//     cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);

//     cudaEventRecord(start);
//     // Launch kernel
//     int blockSize = 128;
//     int numBlocks = N;
//     softmax_forward_kernel2<<<numBlocks, blockSize>>>(d_out, d_inp, N, C);
//     cudaEventRecord(stop);

//     // Wait for the event to complete
//     cudaEventSynchronize(stop);

//     // Calculate milliseconds
//     float gpu_time_ms = 0;
//     cudaEventElapsedTime(&gpu_time_ms, start, stop);

//     // Copy result back to host
//     cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

//     // Cleanup
//     cudaFree(d_out);
//     cudaFree(d_inp);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     // Compare results
//     bool success = compare_results(out_cpu, out_gpu, N, C);
//     std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

//     // Print performance comparison
//     std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
//     std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
//     std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x" << std::endl;

//     // Cleanup
//     free(inp);
//     free(out_cpu);
//     free(out_gpu);

//     return 0;
// }
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

    cudaFree(device_inp);
    cudaFree(device_out);
    free(inp);
    free(out_cpu);
    free(out_gpu);

    return 0;
}
