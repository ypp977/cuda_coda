#include <cuda_runtime.h>

#include <chrono>  // 用于 CPU 计时
#include <iostream>
#include <numeric>
#include <vector>

__inline__ __device__ float block_reduce(float val) {
  const int tid = threadIdx.x;
  const int warpSize = 32;
  int lane = tid % warpSize;
  int warp_id = tid / warpSize;

  // First level: warp reduce using shuffle
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);

  // Only one warp (warp 0) participates in the second reduction
  __shared__ float warpSums[32];
  if (lane == 0) {
    warpSums[warp_id] = val;  // Store warp sum in shared memory
  }
  __syncthreads();

  if (warp_id == 0) {
    val = (tid < blockDim.x / warpSize) ? warpSums[tid] : 0.0f;
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
      val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__global__ void reduce_v4(const float* in, float* out, int n) {
  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    sum += in[i];
  }

  sum = block_reduce(sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = sum;
  }
}

float reduce_cpu(const std::vector<float>& data) {
  float sum = 0.0f;
  for (float val : data) {
    sum += val;
  }
  return sum;
}

const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;  // 1M elements
int main() {
  int num_blocks = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  std::vector<float> h_data(N);

  for (int i = 0; i < N; i++) {
    h_data[i] = 1.0f;  // 简单起见，全部初始化为1.0
  }

  // -------------------------------
  // CPU 计时开始
  auto cpu_start = std::chrono::high_resolution_clock::now();

  float cpu_result = reduce_cpu(h_data);

  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
  // CPU 计时结束
  // -------------------------------

  std::cout << "CPU result: " << cpu_result << std::endl;
  std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;

  float *d_data, *d_result;
  float* d_final_result;
  float gpu_result;

  cudaMalloc(&d_data, N * sizeof(float));
  cudaMalloc(&d_result, num_blocks * sizeof(float));
  cudaMalloc(&d_final_result, 1 * sizeof(float));

  cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  // -------------------------------
  // GPU 计时开始 (CUDA Events)
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  reduce_v4<<<num_blocks, BLOCK_SIZE>>>(d_data, d_result, N);
  reduce_v4<<<1, num_blocks>>>(d_result, d_final_result, num_blocks);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // GPU 计时结束
  // -------------------------------

  std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

  cudaMemcpy(&gpu_result, d_final_result, sizeof(float),
             cudaMemcpyDeviceToHost);
  std::cout << "GPU result: " << gpu_result << std::endl;

  if (abs(cpu_result - gpu_result) < 1e-5) {
    std::cout << "Result verified successfully!" << std::endl;
  } else {
    std::cout << "Result verification failed!" << std::endl;
  }

  // 清理资源
  cudaFree(d_data);
  cudaFree(d_result);
  cudaFree(d_final_result);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}