#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

template <typename tens>
__global__ __forceinline__ __launch_bounds__(256, 32)
void scan(
	tens* __restrict__ A,
	tens* __restrict__ B
) {

	const unsigned int id = (blockIdx.x * 256) + threadIdx.x;
	const unsigned int lane_id = id % 32;
	tens value = B[id];
	tens gate = A[id];

	#pragma unroll
	for (unsigned int i = 1; i <= 32; i *= 2) {
		tens n = __shfl_up_sync(0xffffffff, value, i, 32);
		tens g = __shfl_up_sync(0xffffffff, gate, i, 32);
		if (lane_id >= i) {
			value += gate * n;
			gate *= g;
		}
	}

	B[id] = value;
	A[id] = gate;

}


template <typename tens, typename tens_t>
void myscan(const at::Tensor &Ax, const at::Tensor &Bx) {
	auto stream = at::cuda::getCurrentCUDAStream().stream();
	const auto sizes = Bx.sizes();
	const unsigned int batch = sizes[0];
	const auto strides = Bx.strides();
	const unsigned int batch_stride = strides[0];
	constexpr unsigned int block_size = 256;
	const unsigned int grid_size = (batch_stride * batch) / block_size;

	scan<tens><<<grid_size, block_size, 0, stream>>>(
		reinterpret_cast<tens*>(Ax.data_ptr<tens_t>()),
		reinterpret_cast<tens*>(Bx.data_ptr<tens_t>())
	);
	cudaDeviceSynchronize();
}

std::vector<torch::Tensor> myscan_forward(at::Tensor &Ax, at::Tensor &Bx, const unsigned int GT) {
	const unsigned int D = Ax.size(2);
	const unsigned int d_in = Ax.size(3);
	Ax = Ax.view({-1, GT, 32, D, d_in}).movedim({1, 2, 3, 4}, {3, 4, 1, 2}).contiguous();
	Bx = Bx.view({-1, GT, 32, D, d_in}).movedim({1, 2, 3, 4}, {3, 4, 1, 2}).contiguous();


	if (Bx.scalar_type() == at::ScalarType::BFloat16) {
		myscan<__nv_bfloat16, at::BFloat16>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Half) {
		myscan<__half, at::Half>(Ax, Bx);
	} else if (Bx.scalar_type() == at::ScalarType::Float) {
		myscan<float, float>(Ax, Bx);
	} else {
		TORCH_CHECK(false && "Unsupported dtype");
	}

	return {Bx.movedim({1, 2, 3, 4}, {3, 4, 1, 2}),
		Ax.movedim({1, 2, 3, 4}, {3, 4, 1, 2})
	};

}
