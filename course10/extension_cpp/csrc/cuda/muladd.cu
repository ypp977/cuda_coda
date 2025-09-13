#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace extension_cpp {

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  muladd_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mysgemm_v1_kernel(int M, int N, int K, float alpha, const float *A, const float *B,
                                  float beta, float *C) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;  
  int gy = blockIdx.y * blockDim.y + threadIdx.y;  

  if (gx >= N || gy >= M) return;

  float tmp = 0.0f;
  for (int i = 0; i < K; i++) {
    tmp += A[gy * K + i] * B[i * N + gx]; 
  }
  C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}

at::Tensor mysgemm_v1_cuda(
    const at::Tensor& a,
    const at::Tensor& b, double alpha, double beta) {
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  // M × K @ K × N
  auto M = a.size(0);
  auto N = b.size(1);
  TORCH_CHECK(a.size(1) == b.size(0));
  auto K = a.size(1);
  at::SmallVector<long> c_size{M, N};

  at::Tensor c = at::empty(c_size, a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* c_ptr = c.data_ptr<float>();

  const int BLOCK_SIZE = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((M + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
  printf("M=%ld, N=%ld, K=%ld\n", M, N, K);
  mysgemm_v1_kernel<<<blocks, threads, 0, stream>>>(
      M,
      N,
      K,
      static_cast<float>(alpha),
      a_ptr,
      b_ptr,
      static_cast<float>(beta),
      c_ptr);
  return c;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  mul_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  add_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, result_ptr);
}


// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
  m.impl("mysgemm", &mysgemm_v1_cuda);
}
}
