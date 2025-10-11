#include <cuda_runtime.h>
#include <iostream>

// 定义CUDA kernel函数，在GPU上执行
// 参数:
//   device_out - 指向GPU设备端输出数组的指针
//   device_in  - 指向GPU设备端输入数组的指针(常量，只读)
__global__ void test_shuf_up_sync(int* device_out, const int* device_in)
{
    // 从输入数组中读取当前线程对应位置的数据
    // threadIdx.x 是当前线程在线程块中的索引（0到31）
    int val = device_in[threadIdx.x];

    // 使用 __shfl_up_sync 函数进行warp级别的向上数据交换
    // 参数说明：
    // - 0xFFFFFFFF: 掩码，表示warp中的所有线程都参与操作(32位掩码，每一位对应一个线程)
    // - val: 要交换的数据值，即当前线程的寄存器值
    // - 1: 向上移动的步长，当前线程将获取距离自己向上1个位置的上游线程的值
    // - (默认参数32): warp大小，指定参与操作的线程数量，默认为32
    val = __shfl_up_sync(0xFFFFFFFF, val, 1);

    // 将处理后的值写入输出数组对应位置
    // 每个线程将其处理后的结果写入输出数组中与自己线程ID对应的位置
    device_out[threadIdx.x] = val;
}

// 主函数，程序入口点
int main()
{
    // 定义线程数量为32，正好是一个warp的大小
    int num_threads = 32;

    // 声明主机端（CPU内存）的输入和输出数组
    // host_in: 存储输入数据，大小为32个整数
    // host_out: 存储从GPU返回的输出数据，大小为32个整数
    int host_in[num_threads], host_out[num_threads];

    // 初始化主机端输入数据
    // 遍历数组的每个元素，将其设置为对应的索引值
    // 结果：host_in = [0, 1, 2, 3, ..., 31]
    for (int i = 0; i < num_threads; i++)
    {
        // 每个数组元素被设置为其索引值
        // 例如: host_in[0]=0, host_in[1]=1, ..., host_in[31]=31
        host_in[i] = i;
    }

    // 声明设备端（GPU内存）的指针变量
    // device_out: 指向GPU上输出数据的指针
    // device_in: 指向GPU上输入数据的指针
    int *device_out, *device_in;

    // 在GPU上分配内存空间用于存储输出数据
    // cudaMalloc(指针地址, 分配字节数)
    cudaMalloc(&device_out, num_threads * sizeof(int));

    // 在GPU上分配内存空间用于存储输入数据
    cudaMalloc(&device_in, num_threads * sizeof(int));

    // 将主机端的输入数据复制到设备端(GPU内存)
    // cudaMemcpy(目标地址, 源地址, 数据大小, 复制方向)
    // cudaMemcpyHostToDevice: 从主机复制到设备
    cudaMemcpy(device_in, host_in, num_threads * sizeof(int), cudaMemcpyHostToDevice);

    // 启动CUDA kernel函数在GPU上执行
    // <<<1, num_threads>>> 表示: 启动1个线程块，每个线程块包含num_threads(32)个线程
    // 参数传递: device_out(输出数组), device_in(输入数组)
    test_shuf_up_sync<<<1, num_threads>>>(device_out, device_in);

    // 将GPU上的计算结果复制回主机端
    // cudaMemcpyDeviceToHost: 从设备复制到主机
    cudaMemcpy(host_out, device_out, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果标题信息
    std::cout << "results after __shfl_up_sync:\n";

    // 遍历输出数组并打印每个元素的值
    for (int i = 0; i < num_threads; i++)
    {
        // 输出格式: host_out[索引] = 值
        // 例如: host_out[0] = 0, host_out[1] = 0, host_out[2] = 1, ...
        std::cout << "host_out[" << i << "] = " << host_out[i] << "\n";
    }

    // 释放GPU上的输出数据内存
    cudaFree(device_out);

    // 释放GPU上的输入数据内存
    cudaFree(device_in);

    return 0;
}
