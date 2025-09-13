#include <cuda_runtime.h>
#include <iostream>

// 打印hello world
__global__ void hello_world()
{
    // 打印当前 block  thread 索引
    printf("Current block idx = %d ,current thread idx = %d \n", blockIdx.x, threadIdx.x);
    // 如果是线程 id == 0 则打印 hello world
    if (threadIdx.x == 0)
    {
        printf("Current GPU thread idx: %d ,Hello world!\n", threadIdx.x);
    }
}

int main(int argc, char** argv)
{
    // CPU 打印 hello world
    printf("CPU: Hello world!\n");

    // GPU 打印 hello world
    // GPU 线程块数量
    dim3 block_num = 2;
    // 每个线程块线程数
    dim3 block_size = 10;

    // 调用GPU 打印 hello world
    hello_world<<<block_num, block_size>>>();
    // CPU 等待 GPU 运行完成
    cudaDeviceSynchronize();
    // 检查 CUDA 是否产生错误
    if (cudaGetLastError() != cudaSuccess)
    {
        // 如果有错误，把错误打印出来
        std::cerr << "CUDA err: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        // 返回 -1 表示程序异常结束
        return -1;
    }
    else
    {
        // 如果没有错误，说明 GPU 代码顺利执行
        std::cout << "GPU: Hello world finished!" << std::endl;
        // 返回 0 表示正常退出
        return 0;
    }

    std::cout << "CUP: Hello world finished" << std::endl;
    return 0;
}
