#include <cuda.h>
#include <cuda_runtime.h>

#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "bytes_to_type.h"
#include "conversion.h"

// Parameters for one scan launch. Packs the pointers, strides (in element
// units), dims, and the reverse flag into a single struct passed by value to
// the kernel -- mirrors flash-fft-conv / causal-conv1d's ConvParamsBase. The
// full strides are carried so a future channel-last kernel can reuse this
// struct without a signature change.
struct ScanParams {
    const void *__restrict__ gates;
    const void *__restrict__ tokens;
    void *__restrict__ result;
    int batch_stride;
    int dim_stride;
    int seq_stride;
    int batch;
    int dim;
    int seqlen;
    bool reverse;
};

static void set_scan_params(
    ScanParams &params,
    const at::Tensor &gates,
    const at::Tensor &tokens,
    const at::Tensor &out,
    const bool reverse
) {
    memset(&params, 0, sizeof(params));
    params.gates = gates.data_ptr();
    params.tokens = tokens.data_ptr();
    params.result = out.data_ptr();
    params.batch = tokens.size(0);
    params.dim = tokens.size(1);
    params.seqlen = tokens.size(2);
    params.batch_stride = tokens.stride(0);
    params.dim_stride = tokens.stride(1);
    params.seq_stride = tokens.stride(2);
    params.reverse = reverse;
}

template <typename weight_t, int kNStepsPerThread, int kNThreadsPerWarp, int kNWarpsPerBlock, int kNChunksPerSequence, bool kReverse>
__global__ void scan(ScanParams params) {
    constexpr int S = kNStepsPerThread;
    constexpr bool reverse = kReverse;

    __shared__ float warpLastGate[kNWarpsPerBlock];
    __shared__ float warpLastToken[kNWarpsPerBlock];
    __shared__ float chunkAccGate, chunkAccToken;

    const weight_t *__restrict__ gates = reinterpret_cast<const weight_t *>(params.gates);
    const weight_t *__restrict__ tokens = reinterpret_cast<const weight_t *>(params.tokens);
    weight_t *__restrict__ result = reinterpret_cast<weight_t *>(params.result);

    const int seqoffset = blockIdx.x * params.batch_stride + blockIdx.y * params.dim_stride;
    const int warpId = threadIdx.x / kNThreadsPerWarp;
    const int laneId = threadIdx.x % kNThreadsPerWarp;
    const int chunklen = blockDim.x * S;
    constexpr int kBlockLast = kNWarpsPerBlock - 1;
    constexpr int kWarpLast = kNThreadsPerWarp - 1;
    constexpr int kThreadLast = S - 1;

    using vec_t = typename BytesToType<sizeof(weight_t) * S>::Type;
    const int vecIndex = reverse ? (blockDim.x - 1 - threadIdx.x) : threadIdx.x;

    float accGate[S];
    float accToken[S];

    for (int chunk = 0; chunk < kNChunksPerSequence; chunk++) {
        const int offset = seqoffset + (reverse ? kNChunksPerSequence - 1 - chunk : chunk) * chunklen;

        if (chunk) {
            __syncthreads();
        }

        const vec_t gate_vec = reinterpret_cast<const vec_t *>(gates + offset)[vecIndex];
        const vec_t token_vec = reinterpret_cast<const vec_t *>(tokens + offset)[vecIndex];
        const weight_t *gate_elems = reinterpret_cast<const weight_t *>(&gate_vec);
        const weight_t *token_elems = reinterpret_cast<const weight_t *>(&token_vec);

        #pragma unroll
        for (int i = 0; i < S; ++i) {
            const int c = reverse ? (S - 1 - i) : i;
            const float gate = to_float(gate_elems[c]);
            const float token = to_float(token_elems[c]);
            if (i == 0) {
                if (chunk == 0) {
                    accGate[0] = threadIdx.x == 0 ? 1.0f : gate;
                    accToken[0] = token;
                } else {
                    if (threadIdx.x == 0) {
                        accGate[0] = chunkAccGate * gate;
                        accToken[0] = chunkAccToken * gate + token;
                    } else {
                        accGate[0] = gate;
                        accToken[0] = token;
                    }
                }
            } else {
                accGate[i] = accGate[i - 1] * gate;
                accToken[i] = accToken[i - 1] * gate + token;
            }
        }

        #pragma unroll
        for (int delta = 1; delta < kNThreadsPerWarp; delta *= 2) {
            float prev_gate = __shfl_up_sync(0xffffffff, accGate[kThreadLast], delta);
            float prev_token = __shfl_up_sync(0xffffffff, accToken[kThreadLast], delta);

            if (laneId >= delta) {
                #pragma unroll
                for (int i = 0; i < S; ++i) {
                    accToken[i] = prev_token * accGate[i] + accToken[i];
                    accGate[i] = prev_gate * accGate[i];
                }
            }
        }

        __syncwarp();

        if (laneId == kWarpLast) {
            warpLastGate[warpId] = accGate[kThreadLast];
            warpLastToken[warpId] = accToken[kThreadLast];
        }

        __syncthreads();

        if (warpId == 0) {
            float warpAccGate = (laneId < kNWarpsPerBlock) ? warpLastGate[laneId] : 1.0f;
            float warpAccToken = (laneId < kNWarpsPerBlock) ? warpLastToken[laneId] : 0.0f;

            #pragma unroll
            for (int delta = 1; delta < warpSize; delta *= 2) {
                float prev_gate = __shfl_up_sync(0xffffffff, warpAccGate, delta);
                float prev_token = __shfl_up_sync(0xffffffff, warpAccToken, delta);

                if (laneId >= delta) {
                    warpAccToken = prev_token * warpAccGate + warpAccToken;
                    warpAccGate = prev_gate * warpAccGate;
                }
            }

            if (laneId < kNWarpsPerBlock) {
                warpLastGate[laneId] = warpAccGate;
                warpLastToken[laneId] = warpAccToken;
            }
        }

        __syncthreads();

        vec_t out_vec;
        weight_t *out_elems = reinterpret_cast<weight_t *>(&out_vec);
        #pragma unroll
        for (int i = 0; i < S; ++i) {
            if (warpId > 0) {
                accToken[i] = warpLastToken[warpId - 1] * accGate[i] + accToken[i];
                accGate[i] = warpLastGate[warpId - 1] * accGate[i];
            }
            const int c = reverse ? (S - 1 - i) : i;
            from_float(out_elems[c], accToken[i]);
        }
        reinterpret_cast<vec_t *>(result + offset)[vecIndex] = out_vec;

        if (laneId == kWarpLast && warpId == kBlockLast) {
            chunkAccGate = accGate[kThreadLast];
            chunkAccToken = accToken[kThreadLast];
        }
    }
}

template <typename weight_t>
void warpscan(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
    TORCH_CHECK(tokens.stride(-1) == 1 || tokens.size(-1) == 1);
    TORCH_CHECK(gates.stride(-1) == 1 || gates.size(-1) == 1);

    const int seqlen = tokens.size(2);

    ScanParams params;
    set_scan_params(params, gates, tokens, out, reverse);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(params.batch, params.dim);
    constexpr int kNThreadsPerWarp = 32;
    constexpr int kSmem = 0;

    #define LAUNCH_SCAN(S, W, C)                                                                  \
        do {                                                                                     \
            constexpr int kNStepsPerThread = (S), kNWarpsPerBlock = (W), kNChunksPerSequence = (C); \
            const int kNThreads = seqlen / kNStepsPerThread / kNChunksPerSequence;               \
            if (reverse) {                                                                        \
                scan<weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, true> \
                    <<<grid, kNThreads, kSmem, stream>>>(params);                                \
            } else {                                                                              \
                scan<weight_t, kNStepsPerThread, kNThreadsPerWarp, kNWarpsPerBlock, kNChunksPerSequence, false> \
                    <<<grid, kNThreads, kSmem, stream>>>(params);                                \
            }                                                                                    \
        } while (0)

    if (seqlen == 32) {
        LAUNCH_SCAN(1, 1, 1);
    } else if (seqlen == 64) {
        LAUNCH_SCAN(2, 1, 1);
    } else if (seqlen == 128) {
        LAUNCH_SCAN(1, 4, 1);
    } else if (seqlen == 256) {
        LAUNCH_SCAN(1, 8, 1);
    } else if (seqlen == 512) {
        LAUNCH_SCAN(4, 4, 1);
    } else if (seqlen == 1024) {
        LAUNCH_SCAN(4, 8, 1);
    } else if (seqlen == 2048) {
        LAUNCH_SCAN(4, 16, 1);
    } else if (seqlen == 4096) {
        LAUNCH_SCAN(4, 32, 1);
    } else if (seqlen == 8192) {
        LAUNCH_SCAN(4, 32, 2);
    } else if (seqlen == 16384) {
        LAUNCH_SCAN(4, 32, 4);
    } else if (seqlen == 32768) {
        LAUNCH_SCAN(4, 32, 8);
    } else if (seqlen == 65536) {
        LAUNCH_SCAN(4, 32, 16);
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
    }

    #undef LAUNCH_SCAN
}

at::Tensor
warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse) {
    TORCH_CHECK(tokens.is_cuda());
    TORCH_CHECK(gates.is_cuda());
    TORCH_CHECK(tokens.is_contiguous());
    TORCH_CHECK(gates.is_contiguous());

    if (tokens.scalar_type() == at::ScalarType::BFloat16) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::BFloat16);
        warpscan<__nv_bfloat16>(gates, tokens, out, reverse);
    } else if (tokens.scalar_type() == at::ScalarType::Half) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::Half);
        warpscan<__half>(gates, tokens, out, reverse);
    } else if (tokens.scalar_type() == at::ScalarType::Float) {
        TORCH_CHECK(gates.scalar_type() == at::ScalarType::Float);
        warpscan<float>(gates, tokens, out, reverse);
    } else {
        TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
    }
    return out;
}
