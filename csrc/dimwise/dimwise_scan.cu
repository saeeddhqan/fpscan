
#include "shared.h"


template <typename tens>
__device__ void scan_kernel_small_4t_4096_load(
	const tens *__restrict__ Ax,
	const tens *__restrict__ Bx,
	tens* partial_a,
	tens* partial_b,
	const uint tx,
	uint offset
){
	partial_a[0] = tx == 0 ? (tens) 1.0 : Ax[offset];
	partial_b[0] = Bx[offset];

	offset++;
	tens gate = Ax[offset];
	partial_a[1] = partial_a[0] * gate;
	partial_b[1] = partial_b[0] * gate + Bx[offset];

	offset++;
	gate = Ax[offset];
	partial_a[2] = partial_a[1] * gate;
	partial_b[2] = partial_b[1] * gate + Bx[offset];

	offset++;
	gate = Ax[offset];
	partial_a[3] = partial_a[2] * gate;
	partial_b[3] = partial_b[2] * gate + Bx[offset];
}


template <typename tens>
__global__  void scan_kernel_small_4t_4096(
	const tens *__restrict__ Ax,
	tens *__restrict__ Bx,
	const tens *__restrict__ Bh,
	const tens *__restrict__ Wq,
	const tens *__restrict__ Wk,
	uint batch_stride, uint dim_stride
) {
	uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x * 4;
	const uint lane_id = threadIdx.x & 31;
	tens partial_a[4];
	tens partial_b[4];

	scan_kernel_small_4t_4096_load(Ax, Bx, partial_a, partial_b, threadIdx.x, offset);

	#pragma unroll
	for (int delta = 1; delta < 32; delta *= 2) {
		tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[3], delta); // hard-coded step_thread - 1
		tens prev_token = __shfl_up_sync(0xffffffff, partial_b[3], delta); // hard-coded step_thread - 1

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

	Bx[offset] = partial_b[0];
	Bx[offset + 1] = partial_b[1];
	Bx[offset + 2] = partial_b[2];
	Bx[offset + 3] = partial_b[3];
}


template <typename tens>
__device__ void scan_kernel_small_2t_load(
	const tens *__restrict__ Ax,
	tens * Bx,
	tens* partial_a,
	tens* partial_b,
	const uint tx,
	uint offset
){
	partial_a[0] = tx == 0 ? (tens) 1.0 : Ax[offset];
	partial_b[0] = Bx[offset];

	offset++;
	tens gate = Ax[offset];
	partial_a[1] = partial_a[0] * gate;
	partial_b[1] = partial_b[0] * gate + Bx[offset];
}

template <typename tens>
__global__  void scan_kernel_small_2t(
	const tens *__restrict__ Ax,
	tens *__restrict__ Bx,
	const tens *__restrict__ Bh,
	const tens *__restrict__ Wq,
	const tens *__restrict__ Wk,
	uint batch_stride, uint dim_stride
) {
	uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x * 2;
	const uint lane_id = threadIdx.x & 31;
	tens partial_a[2];
	tens partial_b[2];

	partial_a[0] = threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
	partial_b[0] = Bx[offset];
	tens gate = Ax[offset + 1];
	partial_a[1] = partial_a[0] * gate;
	partial_b[1] = partial_b[0] * gate + Bx[offset + 1];
	// scan_kernel_small_2t_load(Ax, Bx, partial_a, partial_b, threadIdx.x, offset);
	
	#pragma unroll
	for (int delta = 1; delta < 32; delta *= 2) {
		tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[1], delta); // hardcoded step_thread - 1
		tens prev_token = __shfl_up_sync(0xffffffff, partial_b[1], delta); // hardcoded step_thread - 1

		if (lane_id >= delta) {
			// You are having two reads for Ax.
			partial_b[0] = prev_token * partial_a[0] + partial_b[0];
			partial_a[0] = prev_gate * partial_a[0];
			partial_b[1] = prev_token * partial_a[1] + partial_b[1];
			partial_a[1] = prev_gate * partial_a[1];
		}
	}

	Bx[offset] = partial_b[0];
	Bx[offset + 1] = partial_b[1];
}



template<typename tens>
__global__ void scan_kernel_small_1t(
	const tens *__restrict__ Ax,
	tens *__restrict__ Bx,
	const tens *__restrict__ Bh,
	const tens *__restrict__ Wq,
	const tens *__restrict__ Wk,
	uint batch_stride, uint dim_stride
	)
{
	const uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x;
	const uint lane_id = threadIdx.x & 31;
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

	Bx[offset] = partial_b; // think about it!
}



void dimwise_pscan(
	torch::Tensor &Ax,
	torch::Tensor &Bx,
	torch::Tensor &Bh,
	torch::Tensor &Wq,
	torch::Tensor &Wk)
{
	const uint batch_stride = Bx.strides()[0];
	// auto stream = at::cuda::getCurrentCUDAStream().stream();
	const auto sizes = Bx.sizes();
	const uint batch_size = sizes[0];
	const uint dim = sizes[1];
	const uint seqlen = sizes[2];

	dim3 grid(batch_size, dim);
	// torch::Tensor out = torch::empty({batch_size, dim, seqlen}, Bx.options());

	if(seqlen == 32) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_1t<input_t><<<grid, 32>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						); 
				}
			)
		);
	} else if (seqlen == 64) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_1t<input_t><<<grid, 64>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						); 
				}
			)
		);
	} else if (seqlen == 128) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_1t<input_t><<<grid, 128>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else if (seqlen == 256) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_1t<input_t><<<grid, 256>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else if (seqlen == 512) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_1t<input_t><<<grid, 512>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else if (seqlen == 1024) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_2t<input_t><<<grid, 512>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else if (seqlen == 2048) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_2t<input_t><<<grid, 1024>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else if (seqlen == 4096) {
		DISPATCH_FLOAT_AND_HALF_AND_BF16(Ax.scalar_type(), Bx.scalar_type(),
			"dimwise scan",
			([&]
				{ scan_kernel_small_4t_4096<input_t><<<grid, 1024>>>(
						static_cast<input_t *>(Ax.data_ptr()),
						static_cast<input_t *>(Bx.data_ptr()),
						static_cast<input_t *>(Bh.data_ptr()),
						static_cast<input_t *>(Wq.data_ptr()),
						static_cast<input_t *>(Wk.data_ptr()),
						// static_cast<input_t *>(out.data_ptr()),
						batch_stride, seqlen
						);
				}
			)
		);
	} else {
		TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
	}
}
