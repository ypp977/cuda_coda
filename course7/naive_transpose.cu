#include <cuda_runtime.h>
#include <iostream>

// 核函数定义
__global__ void naiveGmem(float *out, float *in, int nx, int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix < nx && iy < ny) {
    out[ix * ny + iy] = in[iy * nx + ix];
  }
}

// 调用核函数的封装函数
void call_naiveGmem(float *d_out, float *d_in, int nx, int ny) {
  dim3 blockSize(2, 2); // 线程块大小
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                (ny + blockSize.y - 1) / blockSize.y);
  naiveGmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

int main() {
  int nx = 4;
  int ny = 4;
  size_t size = nx * ny * sizeof(float);

  // 主机内存分配
  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  // 初始化输入矩阵
  for (int i = 0; i < nx * ny; i++) {
    h_in[i] = float(int(i) % 11);
  }

  // 设备内存分配
  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // 将数据从主机复制到设备
  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  // 调用核函数
  call_naiveGmem(d_out, d_in, nx, ny);

  // 将结果从设备复制回主机
  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      std::cout << h_in[j * nx + i] << " ";
    }
    std::cout << "\n";
  }

  printf("---------------\n");

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      std::cout << h_out[j * nx + i] << " ";
    }
    std::cout << "\n";
  }

  // 释放内存
  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed successfully." << std::endl;

  return 0;
}
