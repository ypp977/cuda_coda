#include <cuda_runtime.h>
#include <iostream>

// 定义CUDA kernel函数，在GPU上并行执行
// 功能：将指定源线程的数据广播给同一个warp中的所有线程
// 参数说明：
//   device_out - 指向GPU设备端输出数组的指针
//   device_in  - 指向GPU设备端输入数组的指针(常量，只读)
//   srcLane    - 源线程的lane ID，指定从哪个线程广播数据
__global__ void test_shuf_broadcast(int* device_out, const int* device_in, const int srcLane)
{
    // 每个线程从输入数组中读取与自己线程ID对应位置的数据
    // threadIdx.x 是当前线程在block中的索引(0-31)
    int val = device_in[threadIdx.x];

    // 使用shuffle同步指令将srcLane线程的值广播给所有线程
    // 参数说明：
    // - 0xFFFFFFFF: 掩码，表示warp中的所有32个线程都参与操作
    // - val: 当前线程要参与广播的值
    // - srcLane: 源线程ID，所有线程都会获取这个线程的值
    // 执行后，所有线程的val变量都等于srcLane线程的原始值
    val = __shfl_sync(0xFFFFFFFF, val, srcLane);

    // 将广播得到的值写入输出数组对应位置
    // 每个线程将相同的广播值写入与自己线程ID对应的位置
    device_out[threadIdx.x] = val;
}

// 主函数，程序入口点
int main()
{
    // 定义常量：线程数量为32（一个warp的大小）和源线程ID为9
    // 这意味着将线程9的值(即数字9)广播给所有线程
    const int num_threads = 32, srcLane = 9;

    // 声明主机端（CPU内存）的输入和输出数组
    // host_in: 存储输入数据，大小为32个整数
    // host_out: 存储从GPU返回的输出数据，大小为32个整数
    int host_in[num_threads], host_out[num_threads];

    // 初始化主机端输入数据
    // 通过循环将每个数组元素设置为其索引值
    // 结果：host_in = [0, 1, 2, 3, ..., 31]
    for (int i = 0; i < num_threads; i++)
    {
        host_in[i] = i;
    }
    // 测试数据
    host_in[9] = 977;

    // 声明设备端（GPU内存）的指针变量
    // device_in: 指向GPU上输入数据的指针
    // device_out: 指向GPU上输出数据的指针
    int *device_in, *device_out;

    // 在GPU上分配内存空间用于存储输入数据
    // cudaMalloc(指针地址, 分配字节数)
    cudaMalloc(&device_in, num_threads * sizeof(int));
    // 在GPU上分配内存空间用于存储输出数据
    cudaMalloc(&device_out, num_threads * sizeof(int));

    // 将主机端的输入数据复制到设备端（GPU内存）
    // cudaMemcpy(目标地址, 源地址, 数据大小, 复制方向)
    // cudaMemcpyHostToDevice: 从主机复制到设备
    cudaMemcpy(device_in, host_in, num_threads * sizeof(int), cudaMemcpyHostToDevice);

    // 启动CUDA kernel函数在GPU上执行
    // <<<1, num_threads>>> 表示: 启动1个线程块，每个线程块包含32个线程
    // 参数传递: device_out(输出), device_in(输入), srcLane(源线程ID=9)
    test_shuf_broadcast<<<1, num_threads>>>(device_out, device_in, srcLane);

    // 将GPU上的计算结果复制回主机端
    // cudaMemcpyDeviceToHost: 从设备复制到主机
    cudaMemcpy(host_out, device_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 在控制台打印广播操作的说明信息
    std::cout << "Broadcasting value from thread " << srcLane << ":\n";

    // 遍历输出数组并打印每个元素的值
    // 由于执行了广播操作，所有元素的值都应该等于线程9的原始值(即9)
    for (int i = 0; i < num_threads; i++)
    {
        // 输出格式: host_out[索引] = 值
        std::cout << "host_out[" << i << "] = " << host_out[i] << "\n";
    }

    // 释放GPU上的输出数据内存
    // cudaFree用于释放之前通过cudaMalloc分配的设备内存
    cudaFree(device_out);
    // 释放GPU上的输入数据内存
    cudaFree(device_in);

    return 0;
}
