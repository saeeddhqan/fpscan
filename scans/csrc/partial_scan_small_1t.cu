

template<typename tens, uint warps_per_block>
__forceinline__ __global__ void scan_forward_small_1t(
	const tens *__restrict__ Ax,
	tens * Bx,
	const uint batch_stride,
	const uint dim_stride
) {
	const uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + threadIdx.x;
	const uint lane_id = threadIdx.x % 32;
	tens partial_a[1]; //= threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
	tens partial_b[1];
	partial_a[0] = threadIdx.x == 0 ? (tens) 1.0 : Ax[offset];
	partial_b[0] = Bx[offset];

	#pragma unroll
	for (uint delta = 1; delta < 32; delta *= 2) {
		tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[0], delta); // hard-coded step_thread - 1
		tens prev_token = __shfl_up_sync(0xffffffff, partial_b[0], delta); // hard-coded step_thread - 1

		if (lane_id >= delta) {
			partial_b[0] = prev_token * partial_a[0] + partial_b[0];
			partial_a[0] = prev_gate * partial_a[0];
		}
	}

	// maybe using another empty tensor to save results is a better idea
	printf("%d: %f\n", threadIdx.x, (float)partial_b[0]);
	Bx[offset] = partial_b[0]; // think about it!
}

