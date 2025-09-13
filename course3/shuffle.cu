#include <cuda_runtime.h>

#include <iostream>

__global__ void test_shuf_broadcast(int *dOutput, const int *dInput,
                                    const int srcLane) {
  int val = dInput[threadIdx.x];
  val = __shfl_sync(0xFFFFFFFF, val, srcLane, 32);
  dOutput[threadIdx.x] = val;
}

int main() {
  const int numThreads = 32;
  const int srcLane = 2;

  // Host arrays
  int hInput[numThreads];
  int hOutput[numThreads];

  for (int i = 0; i < numThreads; ++i) {
    hInput[i] = i;
  }

  int *dInput, *dOutput;

  cudaMalloc(&dInput, numThreads * sizeof(int));
  cudaMalloc(&dOutput, numThreads * sizeof(int));

  cudaMemcpy(dInput, hInput, numThreads * sizeof(int), cudaMemcpyHostToDevice);

  test_shuf_broadcast<<<1, numThreads>>>(dOutput, dInput, srcLane);

  cudaMemcpy(hOutput, dOutput, numThreads * sizeof(int),
             cudaMemcpyDeviceToHost);

  std::cout << "Broadcasting value from thread " << srcLane << ":\n";
  for (int i = 0; i < numThreads; ++i) {
    std::cout << "hOutput[" << i << "] = " << hOutput[i] << "\n";
  }

  cudaFree(dInput);
  cudaFree(dOutput);

  return 0;
}
