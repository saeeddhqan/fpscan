#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>


__forceinline__ __device__ float to_float(const float x) { return x; }
__forceinline__ __device__ float to_float(const __half x) { return __half2float(x); }
__forceinline__ __device__ float to_float(const __nv_bfloat16 x) { return __bfloat162float(x); }

__forceinline__ __device__ void from_float(float &dst, const float x) { dst = x; }
__forceinline__ __device__ void from_float(__half &dst, const float x) { dst = __float2half(x); }
__forceinline__ __device__ void from_float(__nv_bfloat16 &dst, const float x) { dst = __float2bfloat16(x); }
