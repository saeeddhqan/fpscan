#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// How about writing another function that works with sequences that doesn't have chunk?
template <typename tens, uint steps_per_thread, uint warps_per_block, uint chunks_per_seq>
__global__ __forceinline__ void scan_forward(
	const tens* __restrict__ Ax,
	tens* Bx,
	const tens* __restrict__ Bh,
	const tens* __restrict__ Wq,
	const tens* __restrict__ Wk,
	const uint batch_stride,
	const uint dim_stride,
	const uint bh_batch_stride,
	const uint bh_dim_stride
) {
	__shared__ tens warp_q[warps_per_block];
	__shared__ tens warp_k[warps_per_block];
	__shared__ tens warp_v[warps_per_block];
	__shared__ tens warp_r[warps_per_block];
	__shared__ tens chunkAccGate, chunkAccToken;

	const uint seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint warp_id = threadIdx.x / 32;
	const uint lane_id = threadIdx.x % 32;
	const uint chunklen = blockDim.x * steps_per_thread; // thread * steps
	const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * bh_dim_stride + warp_id;
	constexpr uint last_thread = steps_per_thread - 1;
	constexpr uint last_warp = 31;
	constexpr uint last_block = warps_per_block - 1;
	const tens empty_gate = 1.0; //constexpr?
	// const tens empty_score = -1e10; //constexpr?

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
		for (uint delta = 1; delta < 32; delta *= 2) {
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

			if (lane_id >= delta) {
				/*
					Try manuall unrolling for a specific length.
				*/
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
			warp_q[warp_id] = Wq[blockIdx.y] * warp_v[warp_id] + warp_v[warp_id];
			warp_r[warp_id] = warp_v[warp_id];
		}

		__syncthreads();

		if (lane_id == 31 && warp_id && warp_id < last_block) {
			tens score = -1e10;
			uint bid;
			#pragma unroll
			for (uint delta = 0; delta < warp_id; ++delta) {
				tens tmp_score = warp_k[delta] * warp_q[warp_id];
				if (tmp_score > score) {
					score = tmp_score;
					bid = delta;
				}
			}
			warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
		}
		__syncthreads();

		// maybe using another empty tensor to save results is a better idea
		#pragma unroll
		for (uint i = 0; i < steps_per_thread; ++i) {
			if (warp_id > 0) {
				partial_b[i] = partial_b[i] + warp_r[warp_id - 1];
			}
			Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
		}

		if (lane_id == last_warp && warp_id == last_block) {
			chunkAccGate = partial_a[last_thread];
			chunkAccToken = partial_b[last_thread];
		}
	}
}


template <typename tens, uint steps_per_thread, uint warps_per_block>
__global__ __forceinline__ void scan_forward_small(
	const tens* __restrict__ Ax,
	tens* Bx,
	const tens* __restrict__ Bh,
	const tens* __restrict__ Wq,
	const tens* __restrict__ Wk,
	const uint batch_stride,
	const uint dim_stride,
	const uint bh_batch_stride,
	const uint bh_dim_stride
) {
	__shared__ tens warp_q[warps_per_block];
	__shared__ tens warp_k[warps_per_block];
	__shared__ tens warp_v[warps_per_block];
	__shared__ tens warp_r[warps_per_block];

	const uint offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint warp_id = threadIdx.x / 32;
	const uint lane_id = threadIdx.x % 32;
	const uint bh_offset = blockIdx.x * bh_batch_stride + blockIdx.y * bh_dim_stride + warp_id;
	constexpr uint last_thread = steps_per_thread - 1;
	constexpr uint last_block = warps_per_block - 1;
	const tens empty_gate = 1.0; //constexpr?
	// const tens empty_score = -1e10; //constexpr?

	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];


	#pragma unroll
	for (uint i = 0; i < steps_per_thread; ++i) {
		const uint chunk_offset = offset + (threadIdx.x * steps_per_thread + i);
		if (i == 0) {
			partial_a[0] = threadIdx.x == 0 ? empty_gate : Ax[chunk_offset];
			partial_b[0] = Bx[chunk_offset];
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
			// Try manuall unrolling for a specific length.
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
		warp_q[warp_id] = Wq[blockIdx.y] * warp_v[warp_id] + warp_v[warp_id];
		warp_r[warp_id] = warp_v[warp_id];
	}

	__syncthreads();

	if (lane_id == 31 && warp_id && warp_id < last_block) {
		tens score = -1e10;
		uint bid;
		#pragma unroll
		for (uint delta = 0; delta < warp_id; ++delta) {
			tens tmp_score = warp_k[delta] * warp_q[warp_id];
			if (tmp_score > score) {
				score = tmp_score;
				bid = delta;
			}
		}
		warp_r[warp_id] = score * warp_v[bid] + warp_v[warp_id];
	}

	__syncthreads();

	// maybe using another empty tensor to save results is a better idea
	#pragma unroll
	for (uint i = 0; i < steps_per_thread; ++i) {
		if (warp_id > 0) {
			partial_b[i] = partial_b[i] + warp_r[warp_id - 1];
		}
		Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
	}
}





template <typename tens, typename torch_tens>
void
pscan_forward(const at::Tensor &Ax,
			  const at::Tensor &Bx,
			  const at::Tensor &Bh,
			  const at::Tensor &Wq,
			  const at::Tensor &Wk
) {

	const auto strides = Bx.strides();
	const uint batch_stride = strides[0];
	const uint bh_batch_stride = Bh.strides()[0];
	const uint bh_dim_stride = Bh.strides()[1];
	const uint dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const uint batch_size = sizes[0];
	const uint dim = sizes[1];
	const uint seqlen = sizes[2];


	auto stream = at::cuda::getCurrentCUDAStream().stream();
	// {64, 128} x {64, 128} suggested by X.
	// We typically use 4 or 8 warps per thread block
	dim3 grid(batch_size, dim);

	if (seqlen == 32) {
		constexpr int warps_per_block = 1;
		scan_forward_small<tens, 1, warps_per_block><<<grid, 32, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 64) {
		constexpr int warps_per_block = 1;
		scan_forward_small<tens, 2, warps_per_block><<<grid, 32, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 128) {
		constexpr int warps_per_block = 4;
		scan_forward_small<tens, 1, warps_per_block><<<grid, 128, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 256) {
		constexpr int warps_per_block = 8;
		scan_forward_small<tens, 1, warps_per_block><<<grid, 256, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 512) {
		constexpr int warps_per_block = 16;
		scan_forward_small<tens, 1, warps_per_block><<<grid, 512, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 1024) {
		constexpr int warps_per_block = 16;
		scan_forward_small<tens, 2, warps_per_block><<<grid, 512, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 2048) {
		constexpr int warps_per_block = 32;
		scan_forward_small<tens, 2, warps_per_block><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 4096) {
		constexpr int warps_per_block = 32;
		scan_forward_small<tens, 4, warps_per_block><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 8192) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 2><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 16384) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 4><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 32768) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 8><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else if (seqlen == 65536) {
		constexpr int warps_per_block = 32;
		scan_forward<tens, 4, warps_per_block, 16><<<grid, 1024, warps_per_block * sizeof(tens) * 4, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bh.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wq.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Wk.data_ptr<torch_tens>()),
			batch_stride, dim_stride, bh_batch_stride, bh_dim_stride
		);
	} else {
		TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
	}
}

at::Tensor
partial_scan_forward(
		const at::Tensor &Ax,
		const at::Tensor &Bx,
		const at::Tensor &Bh,
		const at::Tensor &Wq,
		const at::Tensor &Wk
) {

	if (Bx.scalar_type() == at::ScalarType::BFloat16) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::BFloat16);
		pscan_forward<__nv_bfloat16, at::BFloat16>(Ax, Bx, Bh, Wq, Wk);
	} else if (Bx.scalar_type() == at::ScalarType::Half) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Half);
		pscan_forward<__half, at::Half>(Ax, Bx, Bh, Wq, Wk);
	} else if (Bx.scalar_type() == at::ScalarType::Float) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Float);
		pscan_forward<float, float>(Ax, Bx, Bh, Wq, Wk);
	} else {
		TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
	}
	return Bx;
}
