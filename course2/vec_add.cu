#include <cstdio>

#define BLOCK_SIZE 16  // 每个 block 包含的线程数
// 3个线程块，1个块里有32个线程。
// 0(blockIdx.x) * 32(blockDim.x) + 0(threadIdx.x)
__global__ void vecAdd(int *A, int *B, int *C, int N) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N) {
    int res = A[i] + B[i];
    printf("A[i] = %d\n", A[i]);
    C[i] = res;
  }
}

int main() {
  int N = 10;  // 大规模数组
  size_t size = N * sizeof(int);

  int *A = (int *)malloc(size);
  int *B = (int *)malloc(size);
  int *C = (int *)malloc(size);

  for (int i = 0; i < N; i++) {
    // cpu当中去初始化的
    A[i] = i;
    B[i] = i * 2;
  }

  int *d_A, *d_B, *d_C;
  // 在GPU上分配显存
  // d_A指向显存的起始位置
  cudaMalloc((void **)&d_A, size);
  cudaMalloc((void **)&d_B, size);
  cudaMalloc((void **)&d_C, size);
  // A是指向cpu内存的指针，d_A是指向gpu内存的指针
  // 将cpu内存中的数据拷贝到gpu内存中
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  printf("array size:%d\n", N);
  printf("thread block:%d\n", numBlocks);
  printf("thread num per block:%d\n", BLOCK_SIZE);
  vecAdd<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    if (C[i] != A[i] + B[i]) {
      printf("Error at index %d: Expected %d, Got %d\n", i, A[i] + B[i], C[i]);
      break;
    }
  }
  printf("Vector addition completed successfully.\n");

  free(A);
  free(B);
  free(C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
