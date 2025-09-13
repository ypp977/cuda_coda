#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>

#define TOL 1e-5f
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int block_row_thread = BN / TN;
  const int block_col_thread = BM / TM;
  const int thread_num = block_row_thread * block_col_thread;

  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int ldg_a_num = BK * BM / thread_num / 4;
  const int ldg_b_num = BK * BN / thread_num / 4;

  int a_tile_row = threadIdx.x / (BK / 4);
  int a_tile_col = threadIdx.x % (BK / 4) * 4;
  int a_tile_stride = BM / ldg_a_num;

  int b_tile_row = threadIdx.x / (BN / 4);
  int b_tile_col = threadIdx.x % (BN / 4) * 4;
  int b_tile_stride = BK / ldg_b_num;

  float accum[TM][TN] = {0.};

  float ldg_a_reg[4 * ldg_a_num] = {0.};

  float a_frag[TM];
  float b_frag[TN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

#pragma unroll
  for (int k = 0; k < K; k += BK) {
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
      int ldg_index = i / a_tile_stride * 4;
      FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
          FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
      As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
      As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
      As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
      As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
      FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
          FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
    }
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int i = 0; i < BK; i++) {
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]);
      }
#pragma unroll
      for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n++) {
          accum[m][n] += a_frag[m] * b_frag[n];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int m = 0; m < TM; m++) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
      ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
      ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
      ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
      ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
      FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
    }
  }
}

#define CEIL_DIV(M, N) ((M) + (N) - 1) / (N)
std::vector<int> generateSizes() { return {4096}; }
int main() {
  int device_id = 7;
  checkCudaError(cudaSetDevice(device_id), "cudaSetDevice failed");
  std::vector<int> sizes = generateSizes();
  // 打开CSV文件
  std::ofstream csv_file("sgemm_benchmark_v4.csv");
  csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;

  for (int N : sizes) {
    std::cout << "Testing size: " << N << std::endl;

    size_t size = N * N * sizeof(float);
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_cublas = (float *)malloc(size);
    float *C_v1 = (float *)malloc(size);

    float *d_A, *d_B, *d_C_v1;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");

    bool out_of_memory = false;

    try {
      // 初始化矩阵 A 和 B
      for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
      }

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");
      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");

      // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

      for (int i = 0; i < warpup_time; ++i) {
        mysgemm_v6<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }

      cudaDeviceSynchronize();
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");

      for (int i = 0; i < repeat_time; ++i) {
        mysgemm_v6<128, 128, 8, 8, 8>
            <<<gridDim, blockDim>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");
      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);

    } catch (...) {
      std::cerr << "Out of memory or error during testing size: " << N
                << std::endl;
      out_of_memory = true;
    }

    if (!out_of_memory) {
      std::cout << "Finished size: " << N << std::endl;
    } else {
      csv_file << N << ",OOM,OOM,0" << std::endl;
    }
  }

  csv_file.close();

  std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
  return 0;
}
