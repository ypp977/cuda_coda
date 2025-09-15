#include <cstdio>

// 定义每个block中包含的线程数
#define BLOCK_SIZE 256

// blockIdx.x : 当前Block在整个grid内的索引(0 ~ gridDim.x-1)
// blockDim.x : 每个Block中的线程数
// threadIdx.x : 线程在当前Block中的索引(0 ~ blockDim.x-1)
__global__ void vec_add(int* input_a, int* input_b, int* output_c, int N)
{
    // 计算当前线程要处理的数据元素下标
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    // 判断是否越界：
    // 线程数通常会 >= N，为了避免访问数组时越界，需要加这个条件判断
    if (i < N)
    {
        // 每个线程只负责处理数组中一个元素
        // 将 input_a[i] 和 input_b[i] 相加，并存储到 output_c[i]
        output_c[i] = input_a[i] + input_b[i];
    }
}

int main()
{
    // 数组大小
    int N = 1 << 20; // 1M数据
    // 数组所占字节数
    size_t size = N * sizeof(int);

    // 在CPU端分配内存
    int* host_a = (int*)malloc(size);
    int* host_b = (int*)malloc(size);
    int* host_c = (int*)malloc(size);

    // 初始化数据
    for (int i = 0; i < N; i++)
    {
        host_a[i] = i;
        host_b[i] = i * 2;
    }
    // 定义指向GPU显存中的分配数组的空间
    int *deviece_a, *deviece_b, *deviece_c;
    // 在GPU显存中分配数组存储空间
    cudaMalloc(&deviece_a, size);
    cudaMalloc(&deviece_b, size);
    cudaMalloc(&deviece_c, size);
    // 从CPU上的数组拷贝数据到GPU显存中
    cudaMemcpy(deviece_a, host_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviece_b, host_b, size, cudaMemcpyHostToDevice);

    // 计算所需Block数量
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("array size :%d\n", N);
    printf("thread block nums :%d\n", num_blocks);
    printf("thread num per block :%d\n", BLOCK_SIZE);

    // 启动核函数
    vec_add<<<num_blocks, BLOCK_SIZE>>>(deviece_a, deviece_b, deviece_c, N);
    cudaError_t vec_add_err = cudaDeviceSynchronize();
    if (vec_add_err != cudaSuccess)
    {
        printf("cuda kernel lanuch error:%s\n", cudaGetErrorString(vec_add_err));
        // 释放CPU端内存
        free(host_a);
        free(host_b);
        free(host_c);
        // 释放GPU端内存
        cudaFree(deviece_a);
        cudaFree(deviece_b);
        cudaFree(deviece_c);

        exit(EXIT_FAILURE);
    }

    // 从 device 拷贝数据到 Host
    cudaMemcpy(host_c, deviece_c, size, cudaMemcpyDeviceToHost);

    // 检查结果是否正确
    for (int i = 0; i < N; i++)
    {
        if (host_c[i] != host_a[i] + host_b[i])
        {
            printf("Error at index %d ,Except %d ,Got %d\n", i, host_a[i] + host_b[i], host_c[i]);
            break;
        }
    }
    printf("Vector add complete successfully!\n");

    // 释放CPU端内存
    free(host_a);
    free(host_b);
    free(host_c);
    // 释放GPU端内存
    cudaFree(deviece_a);
    cudaFree(deviece_b);
    cudaFree(deviece_c);

    return 0;
}
