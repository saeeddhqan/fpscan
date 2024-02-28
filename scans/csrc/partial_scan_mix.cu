



template <typename tens, uint steps_per_thread, uint warps_per_block>
__forceinline__ __global__ void scan_forward_small(
	const tens* __restrict__ Ax,
	tens* Bx,
	const uint batch_stride,
	const uint dim_stride
) {
	const uint offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint lane_id = threadIdx.x % 32;
	const tens empty_gate = 1.0; // constexpr?
	constexpr uint last_thread = steps_per_thread - 1;

	tens partial_a[steps_per_thread];
	tens partial_b[steps_per_thread];

	#pragma unroll
	for (uint i = 0; i < steps_per_thread; ++i) {
		const uint chunk_offset = offset + (threadIdx.x * steps_per_thread + i);
		printf("%d: %f\n", threadIdx.x, (float)Bx[chunk_offset]);
		if (i == 0) {
			partial_a[0] = threadIdx.x == 0 ? empty_gate : Ax[chunk_offset];
			partial_b[0] = Bx[chunk_offset];
		} else {
			tens gate = Ax[chunk_offset];
			partial_a[i] = partial_a[i - 1] * gate;
			partial_b[i] = partial_b[i - 1] * gate + Bx[chunk_offset];
		}
	}
	// printf("%d: %f\n", threadIdx.x, (float)partial_b[0]);
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

	// printf("%d: %f\n", threadIdx.x, (float)partial_b[0]);

	// maybe using another empty tensor to save results is a better idea
	#pragma unroll
	for (uint i = 0; i < steps_per_thread; ++i) {
		Bx[offset + threadIdx.x * steps_per_thread + i] = partial_b[i];
	}
}




// template <typename tens, typename torch_tens>
// void
// pscan_forward(const at::Tensor &Ax,
// 			  const at::Tensor &Bx,
// 			  const at::Tensor &Bh,
// 			  const at::Tensor &Wq,
// 			  const at::Tensor &Wk
// ) {

// }

at::Tensor
partial_scan_forward(
	at::Tensor &Ax,
	at::Tensor &Bx,
	at::Tensor &Bh,
	at::Tensor &Wq,
	at::Tensor &Wk
) {


	using input_t = __half;
	if ((Ax.scalar_type() == at::ScalarType::Half) && (Bx.scalar_type() == at::ScalarType::Half) && (Bh.scalar_type() == at::ScalarType::Half) && (Wq.scalar_type() == at::ScalarType::Half) && (Wk.scalar_type() == at::ScalarType::Half)) {
		using input_t = __half;
	} else if((Ax.scalar_type() == at::ScalarType::BFloat16) && (Bx.scalar_type() == at::ScalarType::BFloat16) && (Bh.scalar_type() == at::ScalarType::BFloat16) && (Wq.scalar_type() == at::ScalarType::BFloat16) && (Wk.scalar_type() == at::ScalarType::BFloat16)){
		using input_t = __nv_bfloat16;
	} else if((Ax.scalar_type() == at::ScalarType::Float) && (Bx.scalar_type() == at::ScalarType::Float) && (Bh.scalar_type() == at::ScalarType::Float) && (Wq.scalar_type() == at::ScalarType::Float) && (Wk.scalar_type() == at::ScalarType::Float)){
		using input_t = float;
	} else {
		// TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
		AT_ERROR("scan not implemented for Ax-type '", toString(Ax.scalar_type()), "', Bx-type '", toString(Bx.scalar_type()), "', Bh-type '", toString(Bh.scalar_type()), "', Wq-type '", toString(Wq.scalar_type()), "', Wk-type '", toString(Wk.scalar_type()), "'");
	}

	CHECK_INPUT(Ax);
	CHECK_INPUT(Bx);
	CHECK_INPUT(Bh);
	CHECK_INPUT(Wq);
	CHECK_INPUT(Wk);


	const auto strides = Bx.strides();
	const uint batch_stride = strides[0];
	const uint dim_stride = strides[1];

	const auto sizes = Bx.sizes();
	const uint batch_size = sizes[0];
	const uint dim = sizes[1];
	const uint seqlen = sizes[2];

	dim3 grid(batch_size, dim);

	if(seqlen == 32) {
		constexpr int warps_per_block = 1;
		scan_forward_small<input_t, 1, 1><<<grid, 32>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 64) {
		constexpr int warps_per_block = 2;
		scan_forward_small_2t<input_t, warps_per_block><<<grid, 32>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 128) {
		constexpr int warps_per_block = 2;
		scan_forward_small_2t<input_t, warps_per_block><<<grid, 64>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 256) {
		constexpr int warps_per_block = 4;
		scan_forward_small_2t<input_t, warps_per_block><<<grid, 128>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 512) {
		constexpr int warps_per_block = 16;
		scan_forward_small_1t<input_t, warps_per_block><<<grid, 512>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 1024) {
		constexpr int warps_per_block = 16;
		scan_forward_small_2t<input_t, warps_per_block><<<grid, 512>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 2048) {
		constexpr int warps_per_block = 32;
		scan_forward_small_2t<input_t, warps_per_block><<<grid, 1024>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else if (seqlen == 4096) {
		constexpr int warps_per_block = 32;
		scan_forward_small_4t_4096<input_t><<<grid, 1024>>>(
			static_cast<input_t *>(Ax.data_ptr()),
			static_cast<input_t *>(Bx.data_ptr()),
			batch_stride, dim_stride
		);
	} else {
		TORCH_CHECK(false && "seqlen must be a power of 2, >= 32, <= 65536");
	}
	// if (Bx.scalar_type() == at::ScalarType::BFloat16) {
	// 	TORCH_CHECK(Ax.scalar_type() == at::ScalarType::BFloat16);
	// 	pscan_forward<__nv_bfloat16, at::BFloat16>(Ax, Bx, Bh, Wq, Wk);
	// } else if (Bx.scalar_type() == at::ScalarType::Half) {
	// 	TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Half);
	// 	pscan_forward<__half, at::Half>(Ax, Bx, Bh, Wq, Wk);
	// } else if (Bx.scalar_type() == at::ScalarType::Float) {
	// 	TORCH_CHECK(Ax.scalar_type() == at::ScalarType::Float);
	// 	pscan_forward<float, float>(Ax, Bx, Bh, Wq, Wk);
	// } else {
	// 	TORCH_CHECK(false && "Unsupported tensor dtype: expecting bfloat16, float16 or float32");
	// }
	return Bx;
}
