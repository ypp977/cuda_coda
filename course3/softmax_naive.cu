// 引入用于计时的chrono库
#include <chrono>
// 引入数学库，主要用于INFINITY常量
#include <cmath>
// 引入标准库函数，如malloc/free
#include <cstdlib>
// 引入输入输出流库
#include <iostream>

// CPU版本的softmax前向传播实现
// 参数:
//   out - 输出数组指针
//   inp - 输入数组指针
//   N - 批处理大小（行数）
//   C - 每行的元素数量（列数）
void softmax_forward_cpu(float* out, const float* inp, int N, int C)
{
    // 遍历每一批数据（每一行）
    for (int i = 0; i < N; i++)
    {
        // 获取当前行的输入数据指针
        const float* inp_row = inp + i * C;
        // 获取当前行的输出数据指针
        float* out_row = out + i * C;

        // 初始化最大值为负无穷，用于数值稳定性
        float max_val = -INFINITY;
        // 遍历当前行的所有元素，找到最大值
        for (int j = 0; j < C; j++)
        {
            // 如果当前元素大于已知最大值，则更新最大值
            if (inp_row[j] > max_val)
            {
                max_val = inp_row[j];
            }
        }

        // 计算softmax的分子部分（exp(x-max)）并累加总和
        float sum = 0.f;
        // 遍历当前行的所有元素
        for (int j = 0; j < C; j++)
        {
            // 计算exp(x-max)，减去最大值防止溢出
            out_row[j] = expf(inp_row[j] - max_val);
            // 累加到总和中
            sum += out_row[j];
        }

        // 计算归一化因子（总和的倒数）
        float norm = 1.f / (float)sum;
        // 遍历当前行的所有元素
        for (int j = 0; j < C; j++)
        {
            // 将每个元素除以总和，完成softmax归一化
            out_row[j] *= norm;
        }
    }
}

// CUDA kernel函数，GPU版本1：每个线程处理一整行
// 参数含义与CPU版本相同
__global__ void softmax_forward_kernel1(float* out, const float* inp, int N, int C)
{
    // 计算当前线程负责处理的行索引
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 确保线程索引在有效范围内
    if (i < N)
    {
        // 获取当前行的输入数据指针
        const float* inp_row = inp + i * C;
        // 获取当前行的输出数据指针
        float* out_row = out + i * C;

        // 初始化最大值为负无穷
        float max_val = -INFINITY;
        // 遍历当前行的所有元素，找到最大值
        for (int j = 0; j < C; j++)
        {
            // 如果当前元素大于已知最大值，则更新最大值
            if (inp_row[j] > max_val)
            {
                max_val = inp_row[j];
            }
        }

        // 计算softmax的分子部分（exp(x-max)）并累加总和
        float sum = 0.f;
        // 遍历当前行的所有元素
        for (int j = 0; j < C; j++)
        {
            // 计算exp(x-max)，减去最大值防止溢出
            out_row[j] = expf(inp_row[j] - max_val);
            // 累加到总和中
            sum += out_row[j];
        }

        // 遍历当前行的所有元素，进行归一化
        for (int j = 0; j < C; j++)
        {
            // 将每个元素除以总和，完成softmax归一化
            out_row[j] /= (float)sum;
        }
    }
}

// 简单结果比较函数，用于验证CPU和GPU结果的一致性
// 参数:
//   cpu - CPU计算结果指针
//   gpu - GPU计算结果指针
//   N - 批处理大小
//   C - 每行元素数
//   epsilon - 误差容忍度，默认为1e-3
bool compare_results(const float* cpu, const float* gpu, int N, int C, float epsilon = 1e-3f)
{
    // 遍历所有元素进行比较
    for (int i = 0; i < N * C; ++i)
    {
        // 如果两个结果的差值超过容忍度
        if (fabs(cpu[i] - gpu[i]) > epsilon)
        {
            // 打印不匹配的信息
            printf("Mismatch at index: %d ,CPU=  %.3f ,GPU= %.3f ,diff= %.9f\n", i, cpu[i], gpu[i],
                   fabs(cpu[i] - gpu[i]));
            // 返回false表示结果不匹配
            return false;
        }
    }
    // 所有元素都匹配，返回true
    return true;
}

// CUDA kernel函数，GPU版本2：使用shared memory优化归约操作
__global__ void softmax_forward_kernel2(float* out, const float* inp, int N, int C)
{
    // 声明外部动态shared memory
    extern __shared__ float shared[];
    // 当前block负责处理的行索引
    int idx = blockIdx.x;
    // 当前线程在block内的索引
    int tid = threadIdx.x;
    // block内的线程总数
    int block_size = blockDim.x;

    // 指向当前行的输入数据
    const float* x = inp + idx * C;

    // 初始化最大值为负无穷
    float max_val = -INFINITY;
    // 每个线程跨步处理多个元素（处理元素索引为tid, tid+block_size, tid+2*block_size...）
    for (int i = tid; i < C; i += block_size)
    {
        // 更新最大值
        max_val = fmaxf(max_val, x[i]);
    }
    // 将每个线程找到的局部最大值存入shared memory
    shared[tid] = max_val;
    // 同步block内所有线程，确保所有线程都完成了写入
    __syncthreads();

    // 使用shared memory进行归约操作找到全局最大值
    // stride每次减半，实现树形归约
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        // 只有索引小于stride的线程参与归约
        if (tid < stride)
        {
            // 比较并保留较大值
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        // 同步确保归约操作完成
        __syncthreads();
    }
    // 此时shared[0]中存储了整行的最大值
    max_val = shared[0];

    // 计算exp(x-max)，每个线程处理多个元素
    for (int i = tid; i < C; i += block_size)
    {
        // 将结果存储到输出数组对应位置
        out[i + idx * C] = expf(x[i] - max_val);
    }

    // 更新x指针指向输出数据
    x = out + idx * C;
    // 初始化总和为0
    float sum = 0.f;
    // 每个线程跨步累加多个元素
    for (int i = tid; i < C; i += block_size)
    {
        // 累加元素值
        sum += x[i];
    }
    // 将每个线程计算的局部和存入shared memory
    shared[tid] = sum;
    // 同步确保所有线程完成写入
    __syncthreads();

    // 使用shared memory进行归约求和
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        // 只有索引小于stride的线程参与归约
        if (tid < stride)
        {
            // 累加操作
            shared[tid] += shared[tid + stride];
        }
        // 同步确保归约操作完成
        __syncthreads();
    }
    // 此时shared[0]中存储了整行的和
    sum = shared[0];

    // 进行归一化操作，每个线程处理多个元素
    for (int i = tid; i < C; i += block_size)
    {
        // 将每个元素除以总和
        out[idx * C + i] = x[i] / sum;
    }
}

// warp级别的归约操作：查找最大值
// 参数: val - 需要归约的值，会被更新为归约结果
__device__ void warpReduceMax(float& val)
{
    // 在warp内进行归约，stride每次减半
    for (int stride = warpSize / 2; stride > 0; stride /= 2)
    {
        // 使用shuffle指令获取其他线程的值并比较，保留较大值
        val = fmaxf(__shfl_down_sync(0xffffffff, val, stride), val);
    }
}

// warp级别的归约操作：求和
// 参数: val - 需要归约的值，会被更新为归约结果
__device__ void warpReduceSum(float& val)
{
    // 在warp内进行归约，stride每次减半
    for (int stride = warpSize / 2; stride > 0; stride /= 2)
    {
        // 使用shuffle指令获取其他线程的值并累加
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
}

// CUDA kernel函数，GPU版本3：使用warp-level shuffle指令优化
__global__ void softmax_forward_kernel3(float* out, const float* inp, int N, int C)
{
    // 当前block负责处理的行索引
    int idx = blockIdx.x;
    // 当前线程在block内的索引
    int tid = threadIdx.x;

    // 指向当前行的输入数据
    const float* x = inp + idx * C;

    // 初始化最大值为负无穷
    float max = -INFINITY;
    // 每个线程跨步处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 更新最大值
        max = fmaxf(x[i], max);
    }

    // 在warp内进行归约找到最大值
    warpReduceMax(max);

    // 将线程0的结果广播到整个block
    max = __shfl_sync(0xffffffff, max, 0);

    // 计算exp(x-max)，每个线程处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 将结果存储到输出数组对应位置
        out[i + idx * C] = expf(x[i] - max);
    }

    // 更新x指针指向输出数据
    x = out + idx * C;

    // 初始化总和为0
    float sum = 0.f;

    // 每个线程跨步累加多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 累加元素值
        sum += x[i];
    }

    // 在warp内进行归约求和
    warpReduceSum(sum);

    // 将线程0的结果广播到整个block
    sum = __shfl_sync(0xffffffff, sum, 0);

    // 进行归一化操作，每个线程处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 将每个元素除以总和
        out[idx * C + i] = x[i] / sum;
    }
}

// CUDA kernel函数，GPU版本4：结合warp和shared memory的混合优化
__global__ void softmax_forward_kernel4(float* out, const float* inp, int N, int C)
{
    // 声明外部动态shared memory
    extern __shared__ float shared[];
    // 当前block负责处理的行索引
    int idx = blockIdx.x;
    // 当前线程在block内的索引
    int tid = threadIdx.x;
    // 当前线程所属的warp索引（每个warp有32个线程）
    int warpId = tid / 32;
    // 当前线程在warp内的索引（0-31）
    int laneId = tid % 32;
    // 每个block内的warp数量
    int warpsPerBlock = blockDim.x / 32;

    // 在shared memory中为最大值分配空间
    float* maxvals = shared;
    // 在shared memory中为和值分配空间（紧接着maxvals之后）
    float* sumvals = &shared[warpsPerBlock];

    // 指向当前行的输入数据
    const float* x = inp + idx * C;

    // 初始化最大值为负无穷
    float maxval = -INFINITY;
    // 每个线程跨步处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 更新局部最大值
        maxval = fmaxf(maxval, x[i]);
    }
    // 在warp内进行归约找到warp内的最大值
    warpReduceMax(maxval);
    // 每个warp的第0个线程（laneId==0）将结果存入shared memory
    if (laneId == 0)
    {
        maxvals[warpId] = maxval;
    }

    // 同步确保所有warp都完成写入
    __syncthreads();

    // block内的第0个线程负责将所有warp的最大值归约为整行最大值
    if (tid == 0)
    {
        // 初始化为第0个warp的最大值
        float val = maxvals[0];
        // 与其他warp的最大值比较
        for (int i = 1; i < warpsPerBlock; i++)
        {
            // 保留较大值
            val = fmaxf(val, maxvals[i]);
        }
        // 将整行最大值存回shared memory
        maxvals[0] = val;
    }
    // 同步确保归约完成
    __syncthreads();

    // 获取整行最大值作为偏移量
    float offset = maxvals[0];

    // 计算exp(x-offset)，每个线程处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 将结果存储到输出数组对应位置
        out[idx * C + i] = expf(x[i] - offset);
    }

    // 更新x指针指向输出数据
    x = out + idx * C;
    // 初始化局部和为0
    float sumval = 0.f;
    // 每个线程跨步累加多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 累加元素值
        sumval += x[i];
    }

    // 在warp内进行归约求和
    warpReduceSum(sumval);

    // 每个warp的第0个线程将结果存入shared memory
    if (laneId == 0)
    {
        sumvals[warpId] = sumval;
    }
    // 同步确保所有warp都完成写入
    __syncthreads();

    // block内的第0个线程负责将所有warp的和值归约为整行和值
    if (tid == 0)
    {
        // 初始化为第0个warp的和值
        float val = sumvals[0];
        // 与其他warp的和值累加
        for (int i = 1; i < warpsPerBlock; i++)
        {
            // 累加
            val += sumvals[i];
        }
        // 将整行和值存回shared memory
        sumvals[0] = val;
    }
    // 同步确保归约完成
    __syncthreads();

    // 获取整行和值
    float sum = sumvals[0];

    // 进行归一化操作，每个线程处理多个元素
    for (int i = tid; i < C; i += blockDim.x)
    {
        // 将每个元素除以总和
        out[idx * C + i] = x[i] / sum;
    }
}

// 主函数
int main()
{
    // 设置批处理大小
    int N = 1024;
    // 设置每行元素数量
    int C = 10240;
    // 计算总元素数量
    size_t num_elements = (size_t)N * C;

    // 在CPU上分配内存
    float* inp = (float*)malloc(num_elements * sizeof(float));
    float* out_cpu = (float*)malloc(num_elements * sizeof(float));
    float* out_gpu = (float*)malloc(num_elements * sizeof(float));

    // 初始化输入数据
    for (int n = 0; n < N; ++n)
    {
        // 遍历每一行
        for (int j = 0; j < C; j++)
        {
            // 用模式数据填充，避免所有值相同
            inp[n * C + j] = float(j % 1000) * 0.001f;
        }
    }

    // 在GPU上分配内存
    float *device_inp, *device_out;
    cudaMalloc((void**)&device_inp, num_elements * sizeof(float));
    cudaMalloc((void**)&device_out, num_elements * sizeof(float));
    // 将输入数据从CPU复制到GPU
    cudaMemcpy(device_inp, inp, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // CPU版本计算及计时
    {
        // 记录开始时间
        auto start_cpu = std::chrono::high_resolution_clock::now();
        // 执行CPU版本softmax
        softmax_forward_cpu(out_cpu, inp, N, C);
        // 记录结束时间
        auto end_cpu = std::chrono::high_resolution_clock::now();
        // 计算执行时间
        std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
        // 打印CPU执行时间
        printf("CPU time: %.3f ms\n", cpu_time.count());
    }
    // GPU版本1计算及计时：一个线程处理一整行
    {
        // 初始化设备和主机输出内存
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 记录开始事件
        cudaEventRecord(start);
        // 设置block大小
        int block_size = 128;
        // 计算需要的block数量
        int block_num = (N + block_size - 1) / block_size;
        // 启动kernel函数
        softmax_forward_kernel1<<<block_num, block_size>>>(device_out, device_inp, N, C);
        // 记录结束事件
        cudaEventRecord(stop);
        // 等待GPU完成计算
        cudaEventSynchronize(stop);

        // 计算GPU执行时间
        float gpu_time_1 = 0;
        cudaEventElapsedTime(&gpu_time_1, start, stop);

        // 将结果从GPU复制到CPU
        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // 比较CPU和GPU结果是否一致
        bool ok = compare_results(out_cpu, out_gpu, N, C);
        // 打印结果和执行时间
        printf("[kernel1] Results match: %s,\n[kernel1] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_1);
        // 销毁CUDA事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU版本2计算及计时：使用shared memory归约
    {
        // 初始化设备和主机输出内存
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 记录开始事件
        cudaEventRecord(start);
        // 设置block大小
        int block_size = 128;
        // 计算需要的block数量（每个block处理一行）
        int block_num = N;
        // 计算shared memory大小
        size_t sharedmem_size = block_size * sizeof(float);
        // 启动kernel函数
        softmax_forward_kernel2<<<block_num, block_size, sharedmem_size>>>(device_out, device_inp,
                                                                           N, C);
        // 记录结束事件
        cudaEventRecord(stop);
        // 等待GPU完成计算
        cudaEventSynchronize(stop);

        // 计算GPU执行时间
        float gpu_time_2 = 0;
        cudaEventElapsedTime(&gpu_time_2, start, stop);

        // 将结果从GPU复制到CPU
        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // 比较CPU和GPU结果是否一致
        bool ok = compare_results(out_cpu, out_gpu, N, C);
        // 打印结果和执行时间
        printf("[kernel2] Results match: %s,\n[kernel2] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_2);
        // 销毁CUDA事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU版本3计算及计时：使用warp-level shuffle优化归约
    {
        // 初始化设备和主机输出内存
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 设置block大小
        int block_size = 128;
        // 计算需要的block数量
        int block_num = N;

        // 记录开始事件
        cudaEventRecord(start);

        // 启动kernel函数
        softmax_forward_kernel3<<<block_num, block_size>>>(device_out, device_inp, N, C);

        // 记录结束事件
        cudaEventRecord(stop);
        // 等待GPU完成计算
        cudaEventSynchronize(stop);

        // 计算GPU执行时间
        float gpu_time_3 = 0;
        cudaEventElapsedTime(&gpu_time_3, start, stop);

        // 将结果从GPU复制到CPU
        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // 比较CPU和GPU结果是否一致
        bool ok = compare_results(out_cpu, out_gpu, N, C);
        // 打印结果和执行时间
        printf("[kernel3] Results match: %s,\n[kernel3] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_3);
        // 销毁CUDA事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // GPU版本4计算及计时：使用warp + shared mem混合优化
    {
        // 初始化设备和主机输出内存
        cudaMemset(device_out, 0, num_elements * sizeof(float));
        memset(out_gpu, 0, num_elements * sizeof(float));

        // 创建CUDA事件用于计时
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 设置block大小
        int block_size = 128;
        // 计算需要的block数量
        int block_num = N;
        // 计算每个block内的warp数量
        int warpsPerBlock = block_size / 32;
        // 计算shared memory大小（需要存储最大值和和值）
        size_t sharedmem_size = 2 * warpsPerBlock * sizeof(float);

        // 记录开始事件
        cudaEventRecord(start);

        // 启动kernel函数
        softmax_forward_kernel4<<<block_num, block_size, sharedmem_size>>>(device_out, device_inp,
                                                                           N, C);
        // 记录结束事件
        cudaEventRecord(stop);
        // 等待GPU完成计算
        cudaEventSynchronize(stop);

        // 计算GPU执行时间
        float gpu_time_4 = 0;
        cudaEventElapsedTime(&gpu_time_4, start, stop);

        // 将结果从GPU复制到CPU
        cudaMemcpy(out_gpu, device_out, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

        // 比较GPU和CPU结果是否一致（注意参数顺序与之前不同）
        bool ok = compare_results(out_gpu, out_cpu, N, C);
        // 打印结果和执行时间
        printf("[kernel4] Results match: %s,\n[kernel4] GPU time: %.3fms\n", (ok ? "YES" : "NO"),
               gpu_time_4);
        // 销毁CUDA事件
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    // 释放GPU内存
    cudaFree(device_inp);
    cudaFree(device_out);
    // 释放CPU内存
    free(inp);
    free(out_cpu);
    free(out_gpu);

    // 程序正常退出
    return 0;
}
