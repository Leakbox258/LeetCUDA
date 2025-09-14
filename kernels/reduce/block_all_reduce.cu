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

// Block All Reduce Sum
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
}

// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {}

//  FP16
//  Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {}

// Block All Reduce Sum: Half
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half *a, float *y, int N) {}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half *a, float *y, int N) {}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *a, float *y,
                                                      int N) {}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *a, float *y,
                                                      int N) {}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *a, float *y,
                                                           int N) {}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half *a, float *y,
                                                           int N) {}

//  BF16
//  Warp Reduce Sum: Half
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16
warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {}

// Block All Reduce Sum: BF16
// grid(N/256), block(256)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16 *a,
                                                      float *y, int N) {}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16 *a, float *y,
                                                     int N) {}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16 *a,
                                                        float *y, int N) {}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16 *a,
                                                       float *y, int N) {}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16 *a,
                                                             float *y, int N) {}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16 *a,
                                                            float *y, int N) {}

//  FP8
//
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) {
  // typedef unsigned char __nv_fp8_storage_t;
  // __half &operator=(const __half_raw &hr);
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {}

template <const int NUM_THREADS = 256 / 16>
__global__ void
block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                 float *y, int N) {}

template <const int NUM_THREADS = 256 / 16>
__global__ void
block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                 float *y, int N) {}

//  INT8
//
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(int8_t *a, int32_t *y,
                                                   int N) {}

template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t *a,
                                                           int32_t *y, int N) {}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define LANUCH_REDUCE_KERNEL(NT, packed_type, acc_type, element_type,          \
                             out_type)                                         \
  block_all_reduce_sum_##packed_type##_##acc_type##_kernel<(NT)>               \
      <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),        \
                        reinterpret_cast<out_type *>(y.data_ptr()), N);

#define DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
  const int NT = (K) / (n_elements);                                           \
  dim3 block(NT);                                                              \
  dim3 grid((S));                                                              \
  switch (NT) {                                                                \
  case 32:                                                                     \
    LANUCH_REDUCE_KERNEL(32, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 64:                                                                     \
    LANUCH_REDUCE_KERNEL(64, packed_type, acc_type, element_type, out_type)    \
    break;                                                                     \
  case 128:                                                                    \
    LANUCH_REDUCE_KERNEL(128, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 256:                                                                    \
    LANUCH_REDUCE_KERNEL(256, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 512:                                                                    \
    LANUCH_REDUCE_KERNEL(512, packed_type, acc_type, element_type, out_type)   \
    break;                                                                     \
  case 1024:                                                                   \
    LANUCH_REDUCE_KERNEL(1024, packed_type, acc_type, element_type, out_type)  \
    break;                                                                     \
  default:                                                                     \
    throw std::runtime_error(                                                  \
        "only support (K)/(n_elements): 32/64/128/256/512/1024");              \
    break;                                                                     \
  }

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
  torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(               \
      torch::Tensor x) {                                                       \
    CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
    auto y_th_type =                                                           \
        (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;           \
    auto options =                                                             \
        torch::TensorOptions().dtype(y_th_type).device(torch::kCUDA, 0);       \
    auto y = torch::zeros({1}, options);                                       \
    const int ndim = x.dim();                                                  \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; ++i) {                                         \
        N *= x.size(i);                                                        \
      }                                                                        \
      dim3 block(1024 / (n_elements));                                         \
      dim3 grid((N + 1024 - 1) / 1024);                                        \
      block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /          \
                                                               (n_elements)>   \
          <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),    \
                            reinterpret_cast<out_type *>(y.data_ptr()), N);    \
    } else {                                                                   \
      const int S = x.size(0);                                                 \
      const int K = x.size(1);                                                 \
      const int N = S * K;                                                     \
      if ((K / (n_elements)) <= 1024) {                                        \
        DISPATCH_REDUCE_KERNEL(K, packed_type, acc_type, element_type,         \
                               n_elements, out_type)                           \
      } else {                                                                 \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
          N *= x.size(i);                                                      \
        }                                                                      \
        dim3 block(1024 / (n_elements));                                       \
        dim3 grid((N + 1024 - 1) / 1024);                                      \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<1024 /        \
                                                                 (n_elements)> \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
      }                                                                        \
    }                                                                          \
    return y;                                                                  \
  }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16x2, f16, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x2, f32, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, bf16, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x2, f32, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
TORCH_BINDING_REDUCE(fp8_e4m3, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t,
                     1, float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16, torch::kFloat8_e4m3fn,
                     __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 1,
                     float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16, torch::kFloat8_e5m2,
                     __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(i8, i32, torch::kInt8, int8_t, 1, int32_t)
TORCH_BINDING_REDUCE(i8x16_pack, i32, torch::kInt8, int8_t, 16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
  TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
