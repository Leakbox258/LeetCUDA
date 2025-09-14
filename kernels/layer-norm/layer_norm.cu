#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// FP32
// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {}

// Block reduce sum/max/min device helper for Layer/RMS Norm/Softmax etc.
// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
}

// Layer Norm: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float g, float b,
                                      int N, int K) {}

// Layer Norm Vec4: x: NxK(K=256<1024), y': NxK, y'=x-mean(x)/std(x) each row
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float g, float b,
                                        int N, int K) {}

// FP16
// Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {}

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f16_f32(half val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {}

#define HALF2_SUM(reg, i)                                                      \
  (((idx + (i)) < N * K) ? ((reg).x + (reg).y) : __float2half(0.0f))

#define HALF2_SUB(reg_y, reg_x)                                                \
  (reg_y).x = (reg_x).x - s_mean;                                              \
  (reg_y).y = (reg_x).y - s_mean;

#define HALF2_VARIANCE(reg, i)                                                 \
  (((idx + (i)) < N * K) ? ((reg).x * (reg).x + (reg).y * (reg).y)             \
                         : __float2half(0.0f))

#define HALF2_LAYER_NORM(reg_y, reg_x, g_, b_)                                 \
  (reg_y).x = __hfma((reg_x).x * s_variance, g_, b_);                          \
  (reg_y).y = __hfma((reg_x).y * s_variance, g_, b_);

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float g, float b,
                                            int N, int K) {}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float g, float b,
                                          int N, int K) {}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float g,
                                                 float b, int N, int K) {}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T1, T2)                                       \
  assert((T1).dim() == (T2).dim());                                            \
  for (int i = 0; i < (T1).dim(); ++i) {                                       \
    if ((T2).size(i) != (T1).size(i)) {                                        \
      throw std::runtime_error("Tensor size mismatch!");                       \
    }                                                                          \
  }

// fp32
#define LANUCH_LAYER_NORM_F32_KERNEL(K)                                        \
  layer_norm_f32_kernel<(K)><<<grid, block>>>(                                 \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32_KERNEL(N, K)                                   \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32_KERNEL(64)                                           \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(128)                                          \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(256)                                          \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32_KERNEL(512)                                          \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32_KERNEL(1024)                                         \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F32x4_KERNEL(K)                                      \
  layer_norm_f32x4_kernel<(K) / 4><<<grid, block>>>(                           \
      reinterpret_cast<float *>(x.data_ptr()),                                 \
      reinterpret_cast<float *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F32x4_KERNEL(N, K)                                 \
  dim3 block((K) / 4);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F32x4_KERNEL(64) break;                                  \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(128) break;                                 \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(256) break;                                 \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F32x4_KERNEL(512) break;                                 \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(1024) break;                                \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(2048) break;                                \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F32x4_KERNEL(4096) break;                                \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*4");             \
    break;                                                                     \
  }

// fp16
#define LANUCH_LAYER_NORM_F16F16_KERNEL(K)                                     \
  layer_norm_f16_f16_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F16_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F16_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F16_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F16_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16F32_KERNEL(K)                                     \
  layer_norm_f16_f32_kernel<(K)>                                               \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16F32_KERNEL(N, K)                                \
  dim3 block((K));                                                             \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16F32_KERNEL(64)                                        \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(128)                                       \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(256)                                       \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16F32_KERNEL(512)                                       \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16F32_KERNEL(1024)                                      \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/256/512/1024");           \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x2F16_KERNEL(K)                                   \
  layer_norm_f16x2_f16_kernel<(K) / 2>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x2F16_KERNEL(N, K)                              \
  dim3 block((K) / 2);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x2F16_KERNEL(2048) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*2");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8F16_KERNEL(K)                                   \
  layer_norm_f16x8_f16_kernel<(K) / 8>                                         \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8F16_KERNEL(N, K)                              \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(64) break;                               \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(128) break;                              \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(256) break;                              \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(512) break;                              \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(1024) break;                             \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(2048) break;                             \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(4096) break;                             \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8F16_KERNEL(8192) break;                             \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(K)                             \
  layer_norm_f16x8_pack_f16_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

#define LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(K)                             \
  layer_norm_f16x8_pack_f32_kernel<(K) / 8>                                    \
      <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),                \
                        reinterpret_cast<half *>(y.data_ptr()), g, b, N, (K));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(N, K)                        \
  dim3 block((K) / 8);                                                         \
  dim3 grid((N));                                                              \
  switch ((K)) {                                                               \
  case 64:                                                                     \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(64) break;                         \
  case 128:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(128) break;                        \
  case 256:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(256) break;                        \
  case 512:                                                                    \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(512) break;                        \
  case 1024:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(1024) break;                       \
  case 2048:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(2048) break;                       \
  case 4096:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(4096) break;                       \
  case 8192:                                                                   \
    LANUCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(8192) break;                       \
  default:                                                                     \
    throw std::runtime_error("only support K: 64/128/.../1024*8");             \
    break;                                                                     \
  }

void layer_norm_f32(torch::Tensor x, torch::Tensor y, float g, float b) {}

void layer_norm_f32x4(torch::Tensor x, torch::Tensor y, float g, float b) {}

void layer_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g, float b) {}

void layer_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g, float b) {}

void layer_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g, float b) {}

void layer_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g,
                               float b) {}

void layer_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g,
                               float b) {}

void layer_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g, float b) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(layer_norm_f16x8_pack_f32)
}
