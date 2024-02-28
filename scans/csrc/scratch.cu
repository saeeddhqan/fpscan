#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// How about writing another function that works with sequences that doesn't have chunk?
template <typename tens, uint steps_per_thread, uint warps_per_block, uint chunks_per_seq>
__global__ __forceinline__ void scan_forward(

) {

}

template <typename tens, typename torch_tens>
void
pscan_forward(const at::Tensor &Ax,
			  const at::Tensor &Bx,
			  const at::Tensor &Bh,
			  const at::Tensor &dBh,
			  const at::Tensor &Wq
) {

}

at::Tensor
partial_scan_forward(
		const at::Tensor &Ax,
		const at::Tensor &Bx,
		const at::Tensor &Bh,
		const at::Tensor &dBh,
		const at::Tensor &Wq
) {

	if (Bx.scalar_type() == at::ScalarType::BFloat16) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::BFloat16);
		pscan_forward<__nv_bfloat16, at::BFloat16>(Ax, Bx, Bh, dBh, Wq);
	} else if (Bx.scalar_type() == at::ScalarType::Half) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Half);
		pscan_forward<__half, at::Half>(Ax, Bx, Bh, dBh, Wq);
	} else if (Bx.scalar_type() == at::ScalarType::Float) {
		TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Float);
		pscan_forward<float, float>(Ax, Bx, Bh, dBh, Wq);
	} else {
		TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
	}
	return Bx;
}
