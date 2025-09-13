#include <cuda_runtime.h>

#include <iostream>

__global__ void test_shuf_down_sync(int *dOutput, const int *dInput) {
  int val = dInput[threadIdx.x];
  val = __shfl_down_sync(0xFFFFFFFF, val, 2, 32);
  dOutput[threadIdx.x] = val;
}

int main() {
  const int numThreads = 32;

  // Host arrays
  int hInput[numThreads];
  int hOutput[numThreads];

  // Initialize input on host
  for (int i = 0; i < numThreads; ++i) {
    hInput[i] = i;  // Each thread gets its own index as value
  }

  int *dInput, *dOutput;

  // Allocate device memory
  cudaMalloc(&dInput, numThreads * sizeof(int));
  cudaMalloc(&dOutput, numThreads * sizeof(int));

  // Copy input data to device
  cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);

  // Launch kernel
  test_shuf_down_sync<<<1, numThreads>>>(dOutput, dInput);

  // Copy result back to host
  cudaMemcpy(hOutput, dOutput, numThreads * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Print results
  std::cout << "Results after __shfl_down_sync(..., ..., 2, 16):\n";
  for (int i = 0; i < numThreads; ++i) {
    std::cout << "hOutput[" << i << "] = " << hOutput[i] << "\n";
  }

  // Free device memory
  cudaFree(dInput);
  cudaFree(dOutput);

  return 0;
}
