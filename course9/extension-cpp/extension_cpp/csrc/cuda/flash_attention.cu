#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename Ty, int kBc = 4, int kBr = 4, int kDim = 4>
__global__ void flash_attention_v2_kernel(
    Ty* Q,
    Ty* K,
    Ty* V,
    Ty* O,
    int seqlen,
    int stride_head,
    Ty smScale) {
  int groupSeq = (seqlen + kBc - 1) / kBc;
  int groupTx = (kDim + kBc - 1) / kBc;
  int groupTy = (kDim + kBr - 1) / kBr;

  __shared__ Ty sQ[kBr][kDim];
  __shared__ Ty sK[kBc][kDim];
  __shared__ Ty sV[kBc][kDim];
  __shared__ Ty sO[kBr][kDim];
  __shared__ Ty sQK[kBr][kBc];
  __shared__ Ty sSafeE[kBr][kBc];
  __shared__ Ty sDenom[kBr];
  __shared__ Ty sMax[kBr];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int base_offset = blockIdx.x * stride_head;
  int row = ty + blockIdx.y * blockDim.y;

  if (row >= seqlen) {
    return;
  }

  Q += base_offset;
  K += base_offset;
  V += base_offset;
  O += base_offset;

  for (int i = 0; i < groupTx; i++) {
    sQ[ty][i * kBc + tx] = Q[row * kDim + i * kBc + tx];
    sO[ty][i * kBc + tx] = 0;
  }

  sMax[ty] = -INFINITY;
  sDenom[ty] = 0;

  for (int j = 0; j < groupSeq; j++) {
    if ((j * kBc + tx) < seqlen) {
      for (int i = 0; i < groupTy; i++) {
        sK[tx][i * kBr + ty] = K[j * kBc * kDim + tx * kDim + i * kBr + ty];
        sV[tx][i * kBr + ty] = V[j * kBc * kDim + tx * kDim + i * kBr + ty];
      }
    }

    __syncthreads();

    Ty sum = 0.f;
    for (int i = 0; i < kDim; i++) {
      sum += sQ[ty][i] * sK[tx][i];
    }
    sQK[ty][tx] = sum * smScale;

    __syncthreads();

    Ty localMax = -INFINITY;
    for (int i = 0; i < kBc; i++) {
      localMax = max(localMax, sQK[ty][i]);
    }
    __syncthreads();
    Ty newMax = max(sMax[ty], localMax);

    sSafeE[ty][tx] = exp(sQK[ty][tx] - newMax);
    __syncthreads();

    Ty localDenom = 0.f;
    for (int i = 0; i < kBc; i++) {
      localDenom += sSafeE[ty][i];
    }
    __syncthreads();

    Ty rescaleOld = exp(sMax[ty] - newMax);
    Ty newDenom = sDenom[ty] * rescaleOld + localDenom;

    for (int i = 0; i < groupTx; i++) {
      sO[ty][i * kBc + tx] = (sO[ty][i * kBc + tx] * rescaleOld);
      for (int k = 0; k < kBc; k++) {
        sO[ty][i * kBc + tx] += sSafeE[ty][k] * sV[k][i * kBc + tx];
      }
    }

    sMax[ty] = newMax;
    sDenom[ty] = newDenom;
    __syncthreads();
  }

  for (int i = 0; i < groupTx; i++) {
    O[row * kDim + i * kBc + tx] = sO[ty][i * kBc + tx] / sDenom[ty];
  }
}

torch::Tensor flash_attention_v2_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  int bs = q.size(0);
  int head = q.size(1);
  int seqlen = q.size(2);
  int dim = q.size(3);
  float sm_scale = 1.f / sqrtf(static_cast<float>(dim));
  int stride_head = seqlen * dim;

  auto out = torch::zeros_like(q);

  const int Br = 4;
  const int Bc = 4;
  int Gc = bs * head;
  int Gr = (seqlen + Br - 1) / Br;
  assert(dim % Bc == 0 && seqlen % Br == 0);

  dim3 grid = dim3(Gc, Gr);
  dim3 block = dim3(Bc, Br);

  using scalar_t = float;
  flash_attention_v2_kernel<scalar_t, Bc, Br, 128><<<grid, block>>>(
      q.data_ptr<scalar_t>(),
      k.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),
      out.data_ptr<scalar_t>(),
      seqlen,
      stride_head,
      sm_scale);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if(err != 0){
    printf("fla error:%d\n", err);
  }
  return out;
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("myfla", &flash_attention_v2_cuda);
}