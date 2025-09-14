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

#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define BLOCK_SIZE 256
#define theta 10000.0f

__global__ void rope_f32_kernel(float *x, float *out, int seq_len, int N) {}

// another index method of rope.
__global__ void rope_f32_v2_kernel(float *x, float *out, int seq_len, int N) {}

__global__ void rope_f32x4_pack_kernel(float *x, float *out, int seq_len,
                                       int N) {}

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

void rope_f32(torch::Tensor x, torch::Tensor out) {}

void rope_f32_v2(torch::Tensor x, torch::Tensor out) {}

void rope_f32x4_pack(torch::Tensor x, torch::Tensor out) {}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(rope_f32)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32_v2)
  TORCH_BINDING_COMMON_EXTENSION(rope_f32x4_pack)
}
