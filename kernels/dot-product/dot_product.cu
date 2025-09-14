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

// Dot Product
// grid(N/256), block(256)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256>
__global__ void dot_prod_f32_f32_kernel(float *a, float *b, float *y, int N) {}

// Dot Product + Vec4
// grid(N/256), block(256/4)
// a: Nx1, b: Nx1, y=sum(elementwise_mul(a,b))
template <const int NUM_THREADS = 256 / 4>
__global__ void dot_prod_f32x4_f32_kernel(float *a, float *b, float *y, int N) {
}

// FP16
// Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {}

template <const int NUM_THREADS = 256>
__global__ void dot_prod_f16_f32_kernel(half *a, half *b, float *y, int N) {}

template <const int NUM_THREADS = 256 / 2>
__global__ void dot_prod_f16x2_f32_kernel(half *a, half *b, float *y, int N) {}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_prod_f16x8_pack_f32_kernel(half *a, half *b, float *y,
                                               int N) {}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_DOT_PROD_KERNEL(NT, packed_type, acc_type, element_type)        \
  dot_prod_##packed_type##_##acc_type##_kernel<(NT)>                           \
      <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),        \
                        reinterpret_cast<element_type *>(b.data_ptr()),        \
                        prod.data_ptr<float>(), N);

#define DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_DOT_PROD_KERNEL(32, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_DOT_PROD_KERNEL(64, packed_type, acc_type, element_type)            \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_DOT_PROD_KERNEL(128, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_DOT_PROD_KERNEL(256, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_DOT_PROD_KERNEL(512, packed_type, acc_type, element_type)           \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_DOT_PROD_KERNEL(1024, packed_type, acc_type, element_type)          \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_DOT_PROD(packed_type, acc_type, th_type, element_type,   \
                               n_elements)                                     \
  torch::Tensor dot_prod_##packed_type##_##acc_type(torch::Tensor a,           \
                                                    torch::Tensor b) {         \
    CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                     \
    CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                     \
    auto options =                                                             \
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0); \
    auto prod = torch::zeros({1}, options);                                    \
    const int ndim = a.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 block(256);                                                         \
      dim3 grid(((N + 256 - 1) / 256) / (n_elements));                         \
      dot_prod_##packed_type##_##acc_type##_kernel<256>                        \
          <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),    \
                            reinterpret_cast<element_type *>(b.data_ptr()),    \
                            prod.data_ptr<float>(), N);                        \
    } else {                                                                   \
      const int S = a.size(0);                                                 \
      const int K = a.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_DOT_PROD_KERNEL(K, packed_type, acc_type, element_type,       \
                                 n_elements)                                   \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= a.size(i);                                                      \
        }                                                                      \
        dim3 block(256);                                                       \
        dim3 grid(((N + 256 - 1) / 256) / (n_elements));                       \
        dot_prod_##packed_type##_##acc_type##_kernel<256>                      \
            <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),  \
                              reinterpret_cast<element_type *>(b.data_ptr()),  \
                              prod.data_ptr<float>(), N);                      \
      }                                                                        \
    }                                                                          \
    return prod;                                                               \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack
TORCH_BINDING_DOT_PROD(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PROD(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PROD(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PROD(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PROD(f16x8_pack, f32, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(dot_prod_f16x8_pack_f32)
}
