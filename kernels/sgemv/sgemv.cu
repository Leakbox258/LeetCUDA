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

//  FP32
//  Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {}

// SGEMV: Warp SGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
__global__ void sgemv_k32_f32_kernel(float *a, float *x, float *y, int M,
                                     int K) {}

// SGEMV: Warp SGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *A, float *x, float *y, int M,
                                     int K) {}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

#define ASSERT_K_IS_MULTIBLE_OF(V)                                             \
  if (K % (V) != 0) {                                                          \
    throw std::runtime_error("K must be multiple of " #V);                     \
  }

#define ASSERT_K_IS_EQUAL_OF(V)                                                \
  if (K != (V)) {                                                              \
    throw std::runtime_error("K must be " #V);                                 \
  }

void sgemv_k32_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(32)
}

void sgemv_k128_f32x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(128)
}

void sgemv_k16_f32(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_EQUAL_OF(16)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k32_f32)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k128_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(sgemv_k16_f32)
}
