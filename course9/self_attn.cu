// main.cu
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "helper.h"

#define CUDA_CHECK(condition)                                          \
  do {                                                                 \
    cudaError_t error = condition;                                     \
    if (error != cudaSuccess) {                                        \
      printf("CUDA_CHECK error in line %d of file %s: %s\n", __LINE__, \
             __FILE__, cudaGetErrorString(cudaGetLastError()));        \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

// #define DEBUG

#ifdef DEBUG
#define DEBUG_BLOCK(expr) \
  do {                    \
    expr                  \
  } while (0)
#else
#define DEBUG_BLOCK(...) \
  do {                   \
  } while (0)
#endif

// -------------------------------
// CUDA Kernels (unchanged)
// -------------------------------

__global__ void naive_nrow_gemm(float *A, float *B, float *C, float a, float b,
                                int M, int N, int K, int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = a * sum + b * C[i * N + j];
    }
  }
}

__global__ void naive_pv(float *P, float *V, float *O, int M, int N,
                         int mBlock) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx *= mBlock;

  int K = M;
  for (int i = idx; i < idx + mBlock; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.f;
      for (int k = 0; k < K; k++) {
        sum += P[i * K + k] * V[k * N + j];
      }
      O[i * N + j] = sum;
    }
  }
}

__global__ void row_softmax(float *input, float *output, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  float max_val = -INFINITY;
  float sum = 0.f;

  for (int i = 0; i < n; i++) {
    if (input[idx * n + i] > max_val) {
      max_val = input[idx * n + i];
    }
  }

  for (int i = 0; i < n; i++) {
    output[idx * n + i] = expf(input[idx * n + i] - max_val);
    sum += output[idx * n + i];
  }

  for (int i = 0; i < n; i++) {
    output[idx * n + i] /= sum;
  }
}

// -------------------------------
// Helper: Read from .bin file
// -------------------------------
bool read_bin(const char *filename, float *h_data, size_t num_elements) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    printf("❌ Failed to open %s\n", filename);
    return false;
  }
  file.read((char *)h_data, num_elements * sizeof(float));
  if (!file) {
    printf("❌ Failed to read data from %s\n", filename);
    file.close();
    return false;
  }
  file.close();
  printf("✅ Loaded %s (%zu elements)\n", filename, num_elements);
  return true;
}

// -------------------------------
// Helper: Write to .bin file
// -------------------------------
bool write_bin(const char *filename, const float *h_data, size_t num_elements) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    printf("❌ Failed to create %s\n", filename);
    return false;
  }
  file.write((const char *)h_data, num_elements * sizeof(float));
  file.close();
  printf("✅ Saved %s (%zu elements)\n", filename, num_elements);
  return true;
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
  int mBlock = 2;
  assert(m % mBlock == 0 && "mBlock should align");

  float sm_scale = 1.f / sqrtf(static_cast<float>(n));
  float *sm_o;
  cudaMalloc((void **)&sm_o, sizeof(float) * m * m);

  dim3 qk_block(m / mBlock, 1, 1);
  naive_nrow_gemm<<<1, qk_block>>>(Q, K, sm_o, sm_scale, 0, m, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError()); printf("== naive QK ==\n");
              print_device_matrix(sm_o, m, m););

  // QK[M, M]
  dim3 sm_block(m, 1, 1);
  row_softmax<<<1, sm_block>>>(sm_o, sm_o, m);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK) ==\n");
              print_device_matrix(sm_o, m, m););

  // QK[M,M] @ V[M, N]
  dim3 qkv_block(m / mBlock, 1, 1);
  naive_pv<<<1, qkv_block>>>(sm_o, V, O, m, n, mBlock);
  cudaDeviceSynchronize();
  DEBUG_BLOCK(CUDA_CHECK(cudaGetLastError());
              printf("== naive softmax(QK)V ==\n");
              print_device_matrix(O, m, n););

  cudaFree(sm_o);
}

// -------------------------------
// Self-Attention with I/O
// -------------------------------
void self_attention_with_io(int m, int n) {
  size_t num_elements = m * n;

  // Host memory
  float *h_Q = new float[num_elements];
  float *h_K = new float[num_elements];
  float *h_V = new float[num_elements];
  float *h_O = new float[num_elements];

  // Read inputs
  read_bin("/home/test_fss/code/cuda_code/course9/Q.bin", h_Q, num_elements);
  read_bin("/home/test_fss/code/cuda_code/course9/K.bin", h_K, num_elements);
  read_bin("/home/test_fss/code/cuda_code/course9/V.bin", h_V, num_elements);

  // Device memory
  float *d_Q, *d_K, *d_V, *d_O;
  CUDA_CHECK(cudaMalloc(&d_Q, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_K, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_V, num_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_O, num_elements * sizeof(float)));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, num_elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  //   print_device_matrix(d_Q, m, n);

  // Run self attention
  self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_O, d_O, num_elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  // print_device_matrix(d_O, m, n);
  // Save output to O_cuda.bin
  write_bin("/home/test_fss/code/cuda_code/course9/O_cuda.bin", h_O,
            num_elements);

  // Cleanup
  delete[] h_Q;
  delete[] h_K;
  delete[] h_V;
  delete[] h_O;
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_O);

  printf("🎉 Self-attention completed. Output saved to O_cuda.bin\n");
}

// -------------------------------
// Entry point
// -------------------------------
int main() {
  const int m = 64;
  const int n = 128;

  printf("🚀 Running self-attention for m=%d, n=%d\n", m, n);
  self_attention_with_io(m, n);

  return 0;
}
