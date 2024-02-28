#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// How about writing another function that works with sequences that doesn't have chunk?
template <typename tens, uint steps_per_thread, uint warps_per_block, uint chunks_per_seq>
__global__ __forceinline__ void scan_forward(
	const tens* __restrict__ Ax,
	tens* Bx,
	const uint batch_stride,
	const uint dim_stride
) {
	__shared__ tens warp_last_gate[warps_per_block];
	__shared__ tens warp_last_token[warps_per_block];
	__shared__ tens chunkAccGate, chunkAccToken;

	const uint seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint warp_id = threadIdx.x >> 5; // x / 32
	const uint lane_id = threadIdx.x & 31; // x % 32
	const uint chunklen = blockDim.x * steps_per_thread; // thread * steps
	constexpr uint last_thread = steps_per_thread - 1;
	constexpr uint last_warp = 31;
	constexpr uint last_block = warps_per_block - 1;
	const tens empty_gate = 1.0; //constexpr?
	// const tens empty_score = -1e10; //constexpr?

	tens warp_empty_token;

	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];

	#pragma unroll
	for (int chunk = 0; chunk < chunks_per_seq; chunk++) {
		const uint offset = seq_offset + chunk * chunklen;

		if (chunk) {
			__syncthreads();
		}

		#pragma unroll
		for (int i = 0; i < steps_per_thread; ++i) {
			const uint chunk_offset = offset + (threadIdx.x * steps_per_thread + i);
			if (i == 0) {
				if (chunk == 0) {
					partial_a[0] = threadIdx.x == 0 ? empty_gate : Ax[chunk_offset];
					partial_b[0] = Bx[chunk_offset];
				} else {
					if (threadIdx.x == 0) {
						tens gate = Ax[chunk_offset];
						partial_a[0] = chunkAccGate * gate;
						partial_b[0] = chunkAccToken *  gate + Bx[chunk_offset];
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
		for (int delta = 1; delta < 32; delta *= 2) {
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

			if (lane_id >= delta) {
				/*
					Try manuall unrolling for a specific length.
				*/
				#pragma unroll
				for (int i = 0; i < steps_per_thread; ++i) {
					partial_b[i] = prev_token * partial_a[i] + partial_b[i];
					partial_a[i] = prev_gate * partial_a[i];

				}
			}
		}

		__syncwarp();


		if (lane_id == last_warp) {// results of previous < thread steps * 32
			warp_last_gate[warp_id] = partial_a[last_thread]; // last A of warp! not the first one!
			warp_last_token[warp_id] = partial_b[last_thread];
			warp_empty_token = warp_last_token[warp_id];
		}

		__syncthreads();


		if (lane_id == 31 && warp_id && warp_id < last_block) {
			tens score = -1e10; // it must be -inf
			uint bid;
			#pragma unroll
			for (int delta = 0; delta < warp_id; delta++) {
				tens tmp_score = warp_last_gate[delta] * warp_empty_token;
				if (tmp_score > score) {
					score = tmp_score;
					bid = delta;
				}
			}

			warp_last_token[warp_id] = score * __shfl_up_sync(0xffffffff, warp_empty_token, bid) + warp_empty_token;		
		}

		// maybe using another empty tensor to save results is a better idea
		#pragma unroll
		for (int i = 0; i < steps_per_thread; ++i) {
			if (warp_id > 0) {
				partial_b[i] = partial_b[i] + warp_last_token[warp_id - 1];
			}
			Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
		}

		if (lane_id == last_warp && warp_id == last_block) {
			chunkAccGate = partial_a[last_thread];
			chunkAccToken = partial_b[last_thread];
		}
	}
}

template <typename tens, typename torch_tens>
void
pscan_forward(const at::Tensor &Ax, const at::Tensor &Bx) {
	const auto strides = Bx.strides();
	const uint batch_stride = strides[0];
	const uint dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const uint batch_size = sizes[0];
	const uint dim = sizes[1];
	const uint seqlen = sizes[2];

	auto stream = at::cuda::getCurrentCUDAStream().stream();
	dim3 grid(batch_size, dim);

	if (seqlen == 32) {
		constexpr int warps_per_block = 1;
		scan_forward<tens, 1, warps_per_block, 1><<<grid, 32, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 64) {
		constexpr int warps_per_block = 1;
		scan_forward<tens, 2, warps_per_block, 1><<<grid, 32, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 128) {
		constexpr int warps_per_block = 4;
		scan_forward<tens, 1, warps_per_block, 1><<<grid, 128, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 256) {
		constexpr int warps_per_block = 8;
		scan_forward<tens, 1, warps_per_block, 1><<<grid, 256, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 512) {
		constexpr int warps_per_block = 16;
		scan_forward<tens, 1, warps_per_block, 1><<<grid, 512, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 1024) {
		constexpr int warps_per_block = 16;
		scan_forward<tens, 2, warps_per_block, 1><<<grid, 512, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 2048) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 2, warps_per_block, 1><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 4096) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 1><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 8192) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 2><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 16384) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 4><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 32768) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 8><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 65536) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 16><<<grid, 1024, warps_per_block * sizeof(tens) * 2, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else {
		TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
	}
}

at::Tensor
partial_scan_forward(const at::Tensor &Ax, const at::Tensor &Bx) {

	if (Bx.scalar_type() == at::ScalarType::BFloat16) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::BFloat16);
		pscan_forward<__nv_bfloat16, at::BFloat16>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Half) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Half);
		pscan_forward<__half, at::Half>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Float) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Float);
		pscan_forward<float, float>(Ax, Bx);
	} else {
		TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
	}
	return Bx;
}

template <typename tens, uint steps_per_thread, uint chunks_per_seq>
__global__ __forceinline__ void scan_backward(
	const tens* __restrict__ Ax,
	tens* Bx,
	const uint batch_stride,
	const uint dim_stride
) {

	const uint seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint lane_id = threadIdx.x % 32;
	const uint chunklen = blockDim.x * steps_per_thread; // thread * steps
	constexpr uint last_thread = steps_per_thread - 1;
	const tens empty_gate = 1.0;


	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];

	#pragma unroll
	for (int chunk = 0; chunk < chunks_per_seq; chunk++) {
		const uint offset = seq_offset + (chunks_per_seq - 1 - chunk) * chunklen;
		#pragma unroll
		for (int i = 0; i < steps_per_thread; ++i) {
			const uint chunk_offset = offset + (chunklen - 1 - (threadIdx.x * steps_per_thread + i));
			if (i == 0) {
				if (chunk == 0) {
					partial_a[0] = threadIdx.x == 0 ? empty_gate : Ax[chunk_offset];
				} 
				partial_b[0] = Bx[chunk_offset];
			} else {
				tens gate = Ax[chunk_offset];
				partial_a[i] = partial_a[i - 1] * gate;
				partial_b[i] = partial_b[i - 1] * gate + Bx[chunk_offset];
			}
		}

		#pragma unroll
		for (int delta = 1; delta < 32; delta *= 2) {
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

			if (lane_id >= delta) {
				#pragma unroll
				for (int i = 0; i < steps_per_thread; ++i) {
					partial_b[i] = prev_token * partial_a[i] + partial_b[i];
					partial_a[i] = prev_gate * partial_a[i];
				}
			}
		}

		#pragma unroll
		for (int i = 0; i < steps_per_thread; ++i) {
			Bx[offset + (chunklen - 1 - (threadIdx.x * steps_per_thread + i))] = partial_b[i];
		}

	}
}


template <typename tens, typename torch_tens>
void
pscan_backward(const at::Tensor &Ax, const at::Tensor &Bx) {
	const auto strides = Bx.strides();
	const uint batch_stride = strides[0];
	const uint dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const uint batch_size = sizes[0];
	const uint dim = sizes[1];
	const uint seqlen = sizes[2];

	auto stream = at::cuda::getCurrentCUDAStream().stream();
	dim3 grid(batch_size, dim);

	if (seqlen == 32) {
		scan_backward<tens, 1, 1><<<grid, 32, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 64) {
		scan_backward<tens, 2, 1><<<grid, 32, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 128) {
		scan_backward<tens, 1, 1><<<grid, 128, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 256) {
		scan_backward<tens, 1, 1><<<grid, 256, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 512) {
		scan_backward<tens, 1, 1><<<grid, 512, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 1024) {
		scan_backward<tens, 2, 1><<<grid, 512, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 2048) {
		scan_backward<tens, 2, 1><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 4096) {
		scan_backward<tens, 4, 1><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 8192) {
		scan_backward<tens, 4, 2><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 16384) {
		scan_backward<tens, 4, 4><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 32768) {
		scan_backward<tens, 4, 8><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 65536) {
		scan_backward<tens, 4, 16><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else {
		TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
	}
}

at::Tensor
partial_scan_backward(const at::Tensor &Ax, const at::Tensor &Bx) {
	if (Bx.scalar_type() == at::ScalarType::BFloat16) {
		pscan_backward<__nv_bfloat16, at::BFloat16>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Half) {
		pscan_backward<__half, at::Half>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Float) {
		pscan_backward<float, float>(Ax, Bx);
	} else {
		TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
	}

	return Bx;
}
