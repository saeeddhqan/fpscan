
#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
// https://github.com/HazyResearch/flash-fft-conv
#define DISPATCH_FLOAT_AND_HALF_AND_BF16(INPUT_TYPE, WEIGHT_TYPE, NAME, ...)                     \
  if ((INPUT_TYPE == at::ScalarType::Half) && (WEIGHT_TYPE == at::ScalarType::Half)) {           \
    using input_t = __half;                                                            \
    using weight_t = __half;                                                           \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::BFloat16) && (WEIGHT_TYPE == at::ScalarType::BFloat16)) {    \
    using input_t = __nv_bfloat16;                                                        \
    using weight_t = __nv_bfloat16;                                                       \
    __VA_ARGS__();                                                                       \
  } else if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Float))  { \
    using input_t = float;                                                               \
    using weight_t = float;                                                              \
    __VA_ARGS__();                                                                       \
  } else {                                                                               \
    AT_ERROR(#NAME, " not implemented for input-type '", toString(INPUT_TYPE), "' and weight-type '", toString(WEIGHT_TYPE), "'"); \
  }

