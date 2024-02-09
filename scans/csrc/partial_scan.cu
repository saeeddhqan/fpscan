#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <typename tens, unsigned int steps_per_thread, unsigned int chunks_per_seq>
__global__ __forceinline__ void scan_forward(
	const tens* __restrict__ Ax,
	tens* Bx,
	const unsigned int batch_stride,
	const unsigned int dim_stride
) {

	const unsigned int seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const unsigned int lane_id = threadIdx.x % 32;
	const unsigned int chunklen = blockDim.x * steps_per_thread; // thread * steps
	constexpr unsigned int last_thread = steps_per_thread - 1;
	const tens empty_gate = 1.0;


	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];

	#pragma unroll
	for (unsigned int chunk = 0; chunk < chunks_per_seq; chunk++) {
		const unsigned int offset = seq_offset + chunk * chunklen;

		#pragma unroll
		for (unsigned int i = 0; i < steps_per_thread; ++i) {
			const unsigned int chunk_offset = offset + (threadIdx.x * steps_per_thread + i);
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
		for (unsigned int delta = 1; delta < 32; delta *= 2) {
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

			if (lane_id >= delta) {
				#pragma unroll
				for (unsigned int i = 0; i < steps_per_thread; ++i) {
					partial_b[i] = prev_token * partial_a[i] + partial_b[i];
					partial_a[i] = prev_gate * partial_a[i];
				}
			}
		}

		#pragma unroll
		for (unsigned int i = 0; i < steps_per_thread; ++i) {
			Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
		}

	}
}

template <typename tens, typename torch_tens>
void
pscan_forward(const at::Tensor &Ax, const at::Tensor &Bx) {
	const auto strides = Bx.strides();
	const unsigned int batch_stride = strides[0];
	const unsigned int dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const unsigned int batch_size = sizes[0];
	const unsigned int dim = sizes[1];
	const unsigned int seqlen = sizes[2];

	auto stream = at::cuda::getCurrentCUDAStream().stream();
	dim3 grid(batch_size, dim);

	if (seqlen == 32) {
		scan_forward<tens, 1, 1><<<grid, 32, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 64) {
		scan_forward<tens, 2, 1><<<grid, 32, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 128) {
		scan_forward<tens, 1, 1><<<grid, 128, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 256) {
		scan_forward<tens, 1, 1><<<grid, 256, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 512) {
		scan_forward<tens, 1, 1><<<grid, 512, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 1024) {
		scan_forward<tens, 2, 1><<<grid, 512, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 2048) {
		scan_forward<tens, 2, 1><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 4096) {
		scan_forward<tens, 4, 1><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 8192) {
		scan_forward<tens, 4, 2><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 16384) {
		scan_forward<tens, 4, 4><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 32768) {
		scan_forward<tens, 4, 8><<<grid, 1024, 0, stream>>>(
			reinterpret_cast<tens*>(Ax.data_ptr<torch_tens>()),
			reinterpret_cast<tens*>(Bx.data_ptr<torch_tens>()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 65536) {
		scan_forward<tens, 4, 16><<<grid, 1024, 0, stream>>>(
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

template <typename tens, unsigned int steps_per_thread, unsigned int chunks_per_seq>
__global__ __forceinline__ void scan_backward(
	const tens* __restrict__ Ax,
	tens* Bx,
	const unsigned int batch_stride,
	const unsigned int dim_stride
) {

	const unsigned int seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const unsigned int lane_id = threadIdx.x % 32;
	const unsigned int chunklen = blockDim.x * steps_per_thread; // thread * steps
	constexpr unsigned int last_thread = steps_per_thread - 1;
	const tens empty_gate = 1.0;


	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];

	#pragma unroll
	for (unsigned int chunk = 0; chunk < chunks_per_seq; chunk++) {
		const unsigned int offset = seq_offset + (chunks_per_seq - 1 - chunk) * chunklen;
		#pragma unroll
		for (unsigned int i = 0; i < steps_per_thread; ++i) {
			const unsigned int chunk_offset = offset + (chunklen - 1 - (threadIdx.x * steps_per_thread + i));
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
		for (unsigned int delta = 1; delta < 32; delta *= 2) {
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[last_thread], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[last_thread], delta);

			if (lane_id >= delta) {
				#pragma unroll
				for (unsigned int i = 0; i < steps_per_thread; ++i) {
					partial_b[i] = prev_token * partial_a[i] + partial_b[i];
					partial_a[i] = prev_gate * partial_a[i];
				}
			}
		}

		#pragma unroll
		for (unsigned int i = 0; i < steps_per_thread; ++i) {
			Bx[offset + (chunklen - 1 - (threadIdx.x * steps_per_thread + i))] = partial_b[i];
		}

	}
}


template <typename tens, typename torch_tens>
void
pscan_backward(const at::Tensor &Ax, const at::Tensor &Bx) {
	const auto strides = Bx.strides();
	const unsigned int batch_stride = strides[0];
	const unsigned int dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const unsigned int batch_size = sizes[0];
	const unsigned int dim = sizes[1];
	const unsigned int seqlen = sizes[2];

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
