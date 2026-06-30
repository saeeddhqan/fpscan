/*
We do not progress Ax for the next chunks...
We Only consider Bx of the previous chunk for the next chunk. I hope you succeed.
We use Bh, which is the hidden state of the previous layer to make k.
We use the current value to make q, and k.
*/

#include "shared.h"



template <typename tens>
__global__  void scan_kernel_large_4t_4096(
    const tens *__restrict__ Ax,
    tens *__restrict__ Bx,
    const tens *__restrict__ Bh,
    const tens *__restrict__ Wq,
    const tens *__restrict__ Wk,
    uint batch_stride, uint dim_stride
) {
    const uint warps_per_block = 32;
    const int bh_batch_stride = warps_per_block * gridDim.y;

    tens warp_q;
    __shared__ tens warp_k[warps_per_block];
    __shared__ tens warp_v[warps_per_block];
    __shared__ tens warp_r[warps_per_block];

    uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x * 4;
    const uint warp_id = threadIdx.x >> 5;
    const uint lane_id = threadIdx.x & 31;
    const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * warps_per_block + warp_id;

    tens partial_a[4];
    tens partial_b[4];

    partial_a[0] = threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
    partial_b[0] = Bx[offset];

    tens gate = Ax[offset + 1];
    partial_a[1] = partial_a[0] * gate;
    partial_b[1] = partial_b[0] * gate + Bx[offset + 1];

    gate = Ax[offset + 2];
    partial_a[2] = partial_a[1] * gate;
    partial_b[2] = partial_b[1] * gate + Bx[offset + 2];

    gate = Ax[offset + 3];
    partial_a[3] = partial_a[2] * gate;
    partial_b[3] = partial_b[2] * gate + Bx[offset + 3];

    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[3], delta);
        tens prev_token = __shfl_up_sync(0xffffffff, partial_b[3], delta);

        if (lane_id >= delta) {
            partial_b[0] = prev_token * partial_a[0] + partial_b[0];
            partial_a[0] = prev_gate * partial_a[0];
            partial_b[1] = prev_token * partial_a[1] + partial_b[1];
            partial_a[1] = prev_gate * partial_a[1];
            partial_b[2] = prev_token * partial_a[2] + partial_b[2];
            partial_a[2] = prev_gate * partial_a[2];
            partial_b[3] = prev_token * partial_a[3] + partial_b[3];
            partial_a[3] = prev_gate * partial_a[3];
        }
    }

    __syncwarp();

    if (lane_id == 31 && warp_id < warps_per_block - 1) {
        warp_k[warp_id] = Bh[bh_offset];
        warp_k[warp_id] = Wk[blockIdx.y] * warp_k[warp_id] + warp_k[warp_id];
        warp_v[warp_id] = partial_b[3];
        warp_q = Wq[blockIdx.y] * warp_v[warp_id] + warp_v[warp_id];
        warp_r[warp_id] = warp_v[warp_id];
    }

    __syncthreads();

    if (lane_id == 31 && warp_id && warp_id < warps_per_block - 1) {
        tens score = -1e10; 
        uint bid; // trouble
        #pragma unroll
        for (uint delta = 0; delta < warp_id; ++delta) {
            tens tmp_score = warp_k[delta] * warp_q;
            if (tmp_score > score) {
                score = tmp_score;
                bid = delta;
            }
        }
        warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
    }
    __syncthreads();

    if (warp_id > 0) {
        partial_b[0] = partial_b[0] + warp_r[warp_id - 1];
        partial_b[1] = partial_b[1] + warp_r[warp_id - 1];
        partial_b[2] = partial_b[2] + warp_r[warp_id - 1];
        partial_b[3] = partial_b[3] + warp_r[warp_id - 1];
    }
    Bx[offset] = partial_b[0];
    Bx[offset + 1] = partial_b[1];
    Bx[offset + 2] = partial_b[2];
    Bx[offset + 3] = partial_b[3];
}


template <typename tens, uint chunks_per_seq>
__global__  void scan_kernel_large_4t_32wpb(
    const tens *__restrict__ Ax,
    tens *__restrict__ Bx,
    const tens *__restrict__ Bh,
    const tens *__restrict__ Wq,
    const tens *__restrict__ Wk,
    uint batch_stride, uint dim_stride
) {
    const int steps_per_thread = 4;
    const int warps_per_block = 32;
    const int bh_dim_stride = warps_per_block * chunks_per_seq;
    const int bh_batch_stride = warps_per_block * chunks_per_seq * gridDim.y;

    tens warp_q;
    __shared__ tens warp_k[warps_per_block];
    __shared__ tens warp_v[warps_per_block];
    __shared__ tens warp_r[warps_per_block];
    __shared__ tens chunk_b;

    const uint seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
    const uint warp_id = threadIdx.x / 32;
    const uint lane_id = threadIdx.x % 32;
    const uint chunklen = blockDim.x * steps_per_thread; // thread * steps
    const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * bh_dim_stride + warp_id;
    constexpr uint last_thread = steps_per_thread - 1;
    constexpr uint last_warp = 31;
    constexpr uint last_block = warps_per_block - 1;
    const tens empty_gate = 1.0; //constexpr?

    tens partial_a[steps_per_thread];
    tens partial_b[steps_per_thread];

    #pragma unroll
    for (uint chunk = 0; chunk < chunks_per_seq; chunk++) {
        const uint offset = seq_offset + chunk * chunklen;

        if (chunk) {
            __syncthreads();
        }

        #pragma unroll
        for (uint i = 0; i < steps_per_thread; ++i) {
            const uint chunk_offset = offset + (threadIdx.x * steps_per_thread + i);
            if (i == 0) {
                if (chunk == 0) {
                    partial_a[0] = threadIdx.x == 0 ? empty_gate : Ax[chunk_offset];
                    partial_b[0] = Bx[chunk_offset];
                } else {
                    if (threadIdx.x == 0) {
                        partial_b[0] = Ax[chunk_offset] * chunk_b + Bx[chunk_offset];
                    } else {
                        partial_a[0] = Ax[chunk_offset];
                        partial_b[0] = Bx[chunk_offset];
                    }
                }
            } else {
                tens gate = Ax[chunk_offset];
                partial_a[i] = partial_a[i - 1] * gate;
                partial_b[i] = partial_b[i - 1] * gate + Bx[chunk_offset];
            }
        }

        #pragma unroll
        for (uint delta = 1; delta < 32; delta *= 2) {
            tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
            tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

            if (lane_id >= delta) {
                #pragma unroll
                for (uint i = 0; i < steps_per_thread; ++i) {
                    partial_b[i] = prev_token * partial_a[i] + partial_b[i];
                    partial_a[i] = prev_gate * partial_a[i];
                }
            }
        }

        __syncwarp();

        if (lane_id == 31 && warp_id < last_block) {
            warp_k[warp_id] = Bh[bh_offset];
            warp_k[warp_id] = Wk[blockIdx.y] * warp_k[warp_id] + warp_k[warp_id];
            warp_v[warp_id] = partial_b[last_thread];
            warp_q = Wq[blockIdx.y] * warp_v[warp_id] + warp_v[warp_id];
            warp_r[warp_id] = warp_v[warp_id];
        }

        __syncthreads();

        if (lane_id == 31 && warp_id && warp_id < last_block) {
            tens score = -1e10;
            uint bid; // big trouble if tmp_score got -inf
            #pragma unroll
            for (int delta = 0; delta < warp_id; ++delta) {
                tens tmp_score = warp_k[delta] * warp_q;
                if (tmp_score > score) {
                    score = tmp_score;
                    bid = delta;
                }
            }
            warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
        }
        __syncthreads();

        #pragma unroll
        for (uint i = 0; i < steps_per_thread; ++i) {
            if (warp_id > 0) {
                partial_b[i] = partial_b[i] + warp_r[warp_id - 1];
            }
            Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
        }

        if (lane_id == last_warp && warp_id == last_block) {
            chunk_b = partial_b[last_thread];
        }
    }
}



template <typename tens, uint warps_per_block>
__global__  void scan_kernel_small_2t(
    const tens *__restrict__ Ax,
    tens *__restrict__ Bx,
    const tens *__restrict__ Bh,
    const tens *__restrict__ Wq,
    const tens *__restrict__ Wk,
    uint batch_stride, uint dim_stride
) {

    const int bh_batch_stride = warps_per_block * gridDim.y;

    tens warp_q;
    __shared__ tens warp_k[warps_per_block];
    __shared__ tens warp_v[warps_per_block];
    __shared__ tens warp_r[warps_per_block];

    uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x * 2;
    const uint warp_id = threadIdx.x >> 5;
    const uint lane_id = threadIdx.x & 31;
    const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * warps_per_block + warp_id;

    tens partial_a[2];
    tens partial_b[2];

    partial_a[0] = threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
    partial_b[0] = Bx[offset];
    tens gate = Ax[offset + 1];
    partial_a[1] = partial_a[0] * gate;
    partial_b[1] = partial_b[0] * gate + Bx[offset + 1];
    
    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[1], delta);
        tens prev_token = __shfl_up_sync(0xffffffff, partial_b[1], delta);

        if (lane_id >= delta) {
            partial_b[0] = prev_token * partial_a[0] + partial_b[0];
            partial_a[0] = prev_gate * partial_a[0];
            partial_b[1] = prev_token * partial_a[1] + partial_b[1];
            partial_a[1] = prev_gate * partial_a[1];
        }
    }

    __syncwarp();

    if (lane_id == 31 && warp_id < warps_per_block - 1) {
        warp_k[warp_id] = Bh[bh_offset];
        warp_k[warp_id] = Wk[blockIdx.y] * warp_k[warp_id] + warp_k[warp_id];
        warp_v[warp_id] = partial_b[1];
        warp_q = Wq[blockIdx.y] * partial_b[1] + partial_b[1];
        warp_r[warp_id] = warp_v[warp_id];
    }

    __syncthreads();

    if (lane_id == 31 && warp_id && warp_id < warps_per_block - 1) {
        tens score = -1e10;
        uint bid; // make sure tmp_score doesn't get -inf
        #pragma unroll
        for (int delta = 0; delta < warp_id; ++delta) {
            tens tmp_score = warp_k[delta] * warp_q;
            if (tmp_score > score) {
                score = tmp_score;
                bid = delta;
            }
        }
        warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
    }
    __syncthreads();

    if (warp_id > 0) {
        partial_b[0] = partial_b[0] + warp_r[warp_id - 1];
        partial_b[1] = partial_b[1] + warp_r[warp_id - 1];
    }

    Bx[offset] = partial_b[0];
    Bx[offset + 1] = partial_b[1];

}




template<typename tens, uint warps_per_block>
__global__ void scan_kernel_small_1t(
    const tens *__restrict__ Ax,
    tens *__restrict__ Bx,
    const tens *__restrict__ Bh,
    const tens *__restrict__ Wq,
    const tens *__restrict__ Wk,
    uint batch_stride, uint dim_stride, uint bh_batch_stride, uint bh_dim_stride
    )
{
//     const int bh_batch_stride = warps_per_block * gridDim.y;

    __shared__ tens warp_k[warps_per_block];
    __shared__ tens warp_v[warps_per_block];
    __shared__ tens warp_r[warps_per_block];
    tens warp_q;

    const uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x;
    const uint warp_id = threadIdx.x / 32; // x / 32
    const uint lane_id = threadIdx.x % 32; // x % 32
    const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * bh_dim_stride + warp_id;
    tens partial_a = threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
    tens partial_b = Bx[offset];

    #pragma unroll
    for (int delta = 1; delta < 32; delta *= 2) {
        tens prev_gate = __shfl_up_sync(0xffffffff, partial_a, delta);
        tens prev_token = __shfl_up_sync(0xffffffff, partial_b, delta);

        if (lane_id >= delta) {
            partial_b = prev_token * partial_a + partial_b;
            partial_a = prev_gate * partial_a;
        }
    }

    __syncwarp();

    if (lane_id == 31 && warp_id < warps_per_block - 1) {
        warp_k[warp_id] = Bh[bh_offset];
        warp_k[warp_id] = Wk[blockIdx.y] * warp_k[warp_id] + warp_k[warp_id];
        warp_v[warp_id] = partial_b;
        warp_q = Wq[blockIdx.y] * partial_b + partial_b;
        // printf("%d: %fq, %fpb, %fw, %fk, %fv\n", warp_id, (float)warp_q, (float)partial_b, (float)Wq[blockIdx.y], (float)warp_k[warp_id], (float)warp_v[warp_id]);
        warp_r[warp_id] = partial_b;
    }

    __syncthreads();

    if (lane_id == 31 && warp_id > 0 && warp_id < warps_per_block - 1) {
        tens score = -1e10;
        int bid; // make sure tmp_score doesn't get -inf or inf, otherwise you are in a big trouble.  
        #pragma unroll
        for (int delta = 0; delta < warp_id; ++delta) {
            tens tmp_score = warp_k[delta] * warp_q;
            if (tmp_score > score) {
                score = tmp_score;
                bid = delta;
            }
        }
        warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
    }

    __syncthreads();

    if (warp_id > 0) {
        partial_b = partial_b + warp_r[warp_id - 1];
    }

    Bx[offset] = partial_b;
}




void dimwise_pscan(
    torch::Tensor &Ax,
    torch::Tensor &Bx,
    torch::Tensor &Bh,
    torch::Tensor &Wq,
    torch::Tensor &Wk)
{
    const auto strides = Bx.strides();
    const uint batch_stride = strides[0]; // maybe using block x, y, z reduces register pressure
    const uint dim_stride = strides[1];
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto sizes = Bx.sizes();
    const uint batch_size = sizes[0];
    const uint dim = sizes[1];
    const uint seqlen = sizes[2];
    const uint bh_batch_stride = Bh.strides()[0];
    const uint bh_dim_stride = Bh.strides()[1];

    dim3 grid(batch_size, dim);
    // torch::Tensor out = torch::empty({batch_size, dim, seqlen}, Bx.options());

    if (seqlen == 64) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_1t<input_t, 2><<<grid, 64, 2 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
                    ); 
                }
            )
        );
    } else if (seqlen == 128) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_1t<input_t, 4><<<grid, 128, 4 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 256) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_1t<input_t, 8><<<grid, 256, 8 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 512) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_1t<input_t, 16><<<grid, 512, 16 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 1024) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_2t<input_t, 16><<<grid, 512, 16 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 2048) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_small_2t<input_t, 32><<<grid, 1024, 32 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    }  else if (seqlen == 4096) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_large_4t_4096<input_t><<<grid, 1024, 32 * sizeof(input_t) * 3>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 8192) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_large_4t_32wpb<input_t, 2><<<grid, 1024, 32 * sizeof(input_t) * 4>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 16384) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_large_4t_32wpb<input_t, 4><<<grid, 1024, 32 * sizeof(input_t) * 4>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 32768) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_large_4t_32wpb<input_t, 8><<<grid, 1024, 32 * sizeof(input_t) * 4>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else if (seqlen == 65536) {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
            "dimwise scan",
            ([&]
                { scan_kernel_large_4t_32wpb<input_t, 16><<<grid, 1024, 32 * sizeof(input_t) * 4>>>(
                        static_cast<input_t *>(Ax.data_ptr()),
                        static_cast<input_t *>(Bx.data_ptr()),
                        static_cast<input_t *>(Bh.data_ptr()),
                        static_cast<input_t *>(Wq.data_ptr()),
                        static_cast<input_t *>(Wk.data_ptr()),
                        batch_stride, dim_stride
                    );
                }
            )
        );
    } else {
        TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
    }
}
