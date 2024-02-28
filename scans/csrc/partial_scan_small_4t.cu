

template <typename tens>
__device__ void scan_forward_small_4t_4096_load(
	const tens *__restrict__ Ax,
	const tens * Bx,
	tens* partial_a,
	tens* partial_b,
	const uint tx,
	const uint offset, 
	uint chunk_offset
){
	partial_a[0] = tx == 0 ? (tens) 1.0 : Ax[offset];
	partial_b[0] = Bx[offset];

	tens gate = Ax[chunk_offset];

	partial_a[1] = partial_a[0] * gate;
	partial_b[1] = partial_b[0] * gate + Bx[chunk_offset];

	chunk_offset++;
	gate = Ax[chunk_offset];

	partial_a[2] = partial_a[1] * gate;
	partial_b[2] = partial_b[1] * gate + Bx[chunk_offset];

	chunk_offset++;
	gate = Ax[chunk_offset];

	partial_a[3] = partial_a[2] * gate;
	partial_b[3] = partial_b[2] * gate + Bx[chunk_offset];
}

template <typename tens>
__global__  void scan_forward_small_4t_4096(
	const tens *__restrict__ Ax,
	tens * Bx,
	const uint batch_stride,
	const uint dim_stride
) {
	const uint offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	const uint chunk_offset = offset + threadIdx.x * 4; // remove p
	const uint lane_id = threadIdx.x % 32;
	tens partial_a[4];
	tens partial_b[4];

	scan_forward_small_4t_4096_load(Ax, Bx, partial_a, partial_b, threadIdx.x, offset, chunk_offset);

	#pragma unroll
	for (uint delta = 1; delta < 32; delta *= 2) {
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

	Bx[chunk_offset] = partial_b[0];
	Bx[chunk_offset + 1] = partial_b[1];
	Bx[chunk_offset + 2] = partial_b[2];
	Bx[chunk_offset + 3] = partial_b[3];
}

