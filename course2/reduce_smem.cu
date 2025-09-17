#include <iostream>

// ===============================
// 使用全局内存归约的核函数
// ===============================
__global__ void reduce_global_mem(int* g_idata, int* g_odata, unsigned int n)
{
    // 当前线程在 block 内的索引
    unsigned int tid = threadIdx.x;
    // 当前线程在全局数组中的起始位置（block 对应的全局数据段）
    unsigned int base = blockIdx.x * blockDim.x;
    int* idata = g_idata + base;

    // 如果越界（数据不足 blockSize 个），就把当前元素置 0，避免乱加
    if (base + tid >= n)
    {
        idata[tid] = 0;
    }
    __syncthreads(); // 确保所有线程完成初始化

    // 树形归约：每一步把后半部分数据加到前半部分
    // 当 blockSize >= 1024，前 512 个线程把后 512 的值加进来
    if (blockDim.x >= 1024 && tid < 512)
    {
        idata[tid] += (base + tid + 512 < n) ? idata[tid + 512] : 0;
    }
    __syncthreads();

    // 当 blockSize >= 512，前 256 个线程把后 256 的值加进来
    if (blockDim.x >= 512 && tid < 256)
    {
        idata[tid] += (base + tid + 256 < n) ? idata[tid + 256] : 0;
    }
    __syncthreads();

    // 当 blockSize >= 256，前 128 个线程把后 128 的值加进来
    if (blockDim.x >= 256 && tid < 128)
    {
        idata[tid] += (base + tid + 128 < n) ? idata[tid + 128] : 0;
    }
    __syncthreads();

    // 当 blockSize >= 128，前 64 个线程把后 64 的值加进来
    if (blockDim.x >= 128 && tid < 64)
    {
        idata[tid] += (base + tid + 64 < n) ? idata[tid + 64] : 0;
    }
    __syncthreads();

    // Warp 内规约（最后 32 个线程用 shuffle 指令加速）
    if (tid < 32)
    {
        int val = idata[tid] + idata[tid + 32]; // 先合并 64 → 32
        // __shfl_down_sync 用于 warp 内线程之间交换寄存器值
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // 只有 tid=0 的线程写回 block 的部分和
        if (tid == 0)
        {
            g_odata[blockIdx.x] = val;
        }
    }
}

// ===============================
// 使用共享内存归约的核函数
// ===============================
__global__ void reduce_shared_mem(int* g_idata, int* g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + tid;

    // 声明动态共享内存
    extern __shared__ int shared_mem[];

    // 每个线程把一个元素读到共享内存（越界填 0）
    shared_mem[tid] = (gid < n) ? g_idata[gid] : 0;
    __syncthreads();

    // 树形规约：每次都在共享内存里进行
    if (blockDim.x >= 1024 && tid < 512)
    {
        shared_mem[tid] += shared_mem[tid + 512];
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
    {
        shared_mem[tid] += shared_mem[tid + 256];
    }
    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
    {
        shared_mem[tid] += shared_mem[tid + 128];
    }
    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
    {
        shared_mem[tid] += shared_mem[tid + 64];
    }
    __syncthreads();

    // warp 内规约（共享内存已经缩到 64 个元素）
    if (tid < 32)
    {
        int val = shared_mem[tid] + shared_mem[tid + 32];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (tid == 0)
        {
            g_odata[blockIdx.x] = val;
        }
    }
}

// ===============================
// 主函数
// ===============================
int main()
{
    const int N = 1 << 25;
    size_t size = N * sizeof(int); // 输入数组字节数
    int* host_idata = new int[N];  // host 输入数组
    int* host_odata1 = new int[N]; // host 保存 global 部分和
    int* host_odata2 = new int[N]; // host 保存 shared 部分和

    // 初始化输入数组，所有元素设为 1
    for (int i = 0; i < N; i++)
    {
        host_idata[i] = 1;
    }

    const int blocksize = 1024;
    const int blocknum = (N + blocksize - 1) / blocksize; // block 数量 = N / blockSize 向上取整

    // 分配 device 内存
    int *device_idata1, *device_idata2, *device_odata1, *device_odata2;
    cudaMalloc(&device_idata1, size);
    cudaMalloc(&device_idata2, size);
    cudaMalloc(&device_odata1, blocknum * sizeof(int));
    cudaMalloc(&device_odata2, blocknum * sizeof(int));

    // 把输入数组拷贝到 device
    cudaMemcpy(device_idata1, host_idata, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_idata2, host_idata, size, cudaMemcpyHostToDevice);

    // 预热 kernel（运行几次让 GPU “热身”，避免第一次运行不稳定）
    for (int i = 0; i < 5; i++)
    {
        reduce_global_mem<<<blocknum, blocksize>>>(device_idata1, device_odata1, N);
        reduce_shared_mem<<<blocknum, blocksize, blocksize * sizeof(int)>>>(device_idata2,
                                                                            device_odata2, N);
    }

    // 再拷贝一次原始数据（因为 global 内存版会写回 g_idata）
    cudaMemcpy(device_idata1, host_idata, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_idata2, host_idata, size, cudaMemcpyHostToDevice);

    // 创建事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ====== 测量全局内存归约 ======
    cudaEventRecord(start);
    reduce_global_mem<<<blocknum, blocksize>>>(device_idata1, device_odata1, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float g_ms = 0;
    cudaEventElapsedTime(&g_ms, start, stop);
    cudaMemcpy(host_odata1, device_odata1, blocknum * sizeof(int), cudaMemcpyDeviceToHost);

    // ====== 测量共享内存归约 ======
    cudaEventRecord(start);
    reduce_shared_mem<<<blocknum, blocksize, blocksize * sizeof(int)>>>(device_idata2,
                                                                        device_odata2, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float s_ms = 0;
    cudaEventElapsedTime(&s_ms, start, stop);
    cudaMemcpy(host_odata2, device_odata2, blocknum * sizeof(int), cudaMemcpyDeviceToHost);

    // CPU 端累加所有 block 的部分和，得到最终结果
    long long global_res = 0, shared_res = 0;
    for (int i = 0; i < blocknum; i++)
    {
        global_res += host_odata1[i];
        shared_res += host_odata2[i];
    }

    printf("time for reduce global memory: %.3f ms; sum is %lld \n", g_ms, global_res);
    printf("time for reduce shared memory: %.3f ms; sum is %lld\n", s_ms, shared_res);

    return 0;
}
