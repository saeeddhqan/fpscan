template <typename tens, uint chunks_per_seq>
__global__  void scan_kernel_large_4t_32wpb(
	const tens *__restrict__ Ax,
	tens *__restrict__ Bx,
	const tens *__restrict__ Bh,
	const tens *__restrict__ Wq,
	const tens *__restrict__ Wk,
	uint batch_stride, uint dim_stride
) {
	__shared__ tens warp_q[32]; // -
	__shared__ tens warp_k[32]; // -
	__shared__ tens warp_v[32]; // -
	__shared__ tens warp_r[32]; // no
	__shared__ tens chunkAccGate, chunkAccToken;

	// const uint seq_offset = blockIdx.x * batch_stride + blockIdx.y * dim_stride;
	// const uint warp_id = threadIdx.x >> 5; // x / 32
	// const uint lane_id = threadIdx.x & 31; // x % 32
	// const uint chunklen = blockDim.x * 4; // thread * steps
	// const uint bh_offset = blockIdx.x * (32 * blockIdx.y) + blockIdx.y * (blockDim.x * chunks_per_seq * 4) + warp_id;
	// blockIdx.x * bh_batch_stride + blockIdx.y * bh_dim_stride + warp_id;
	// const tens empty_gate = 1.0; //constexpr? or delete?
	// const tens empty_score = -1e10; //constexpr?

	tens partial_a[4];
	tens partial_b[4];

	#pragma unroll
	for (uint chunk = 0; chunk < chunks_per_seq; chunk++) {
		const uint offset = (blockIdx.x * batch_stride + blockIdx.y * dim_stride) + chunk * blockDim.x * 4;
		
		if (chunk) {
			__syncthreads();
		}

		#pragma unroll
		for (uint i = 0; i < 4; ++i) {
			const uint chunk_offset = offset + (threadIdx.x * 4 + i);
			if (i == 0) {
				if (chunk == 0) {
					partial_a[0] = threadIdx.x == 0 ? (tens) 1.0 : Ax[chunk_offset];
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
			tens prev_gate = __shfl_up_sync(0xffffffff, partial_a[3], delta);
			tens prev_token = __shfl_up_sync(0xffffffff, partial_b[3], delta);
			if (threadIdx.x & 31 >= delta) {
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

		if (threadIdx.x & 31 == 31 && threadIdx.x >> 5 < 31) {
			warp_k[threadIdx.x >> 5] = Bh[blockIdx.x * (32 * blockIdx.y) + blockIdx.y * (blockDim.x * chunks_per_seq * 4) + threadIdx.x >> 5];
			// warp_k[threadIdx.x >> 5] = Bh[0];
			warp_k[threadIdx.x >> 5] = Wk[blockIdx.y] * warp_k[threadIdx.x >> 5] + warp_k[threadIdx.x >> 5];
			warp_v[threadIdx.x >> 5] = partial_b[3];
			warp_q[threadIdx.x >> 5] = Wq[blockIdx.y] * warp_v[threadIdx.x >> 5] + warp_v[threadIdx.x >> 5];
			warp_r[threadIdx.x >> 5] = warp_v[threadIdx.x >> 5];
		}
	
		warp_r[0] = (tens) 1.0; // to reduce divergence

		__syncthreads();

		if (threadIdx.x & 31 == 31 && threadIdx.x >> 5 && threadIdx.x >> 5 < 31) {
			tens score = -1e10;
			uint bid;
			#pragma unroll
			for (uint delta = 0; delta < threadIdx.x >> 5; ++delta) {
				tens tmp_score = warp_k[delta] * warp_q[threadIdx.x >> 5];
				if (tmp_score > score) {
					score = tmp_score;
					bid = delta;
				}
			}
			warp_r[threadIdx.x >> 5 + 1] = score * warp_v[bid] + warp_v[threadIdx.x >> 5];
		}

		__syncthreads();

		Bx[offset + threadIdx.x * 4] = partial_b[0] + warp_r[threadIdx.x >> 5];
		Bx[offset + threadIdx.x * 4 + 1] = partial_b[1] + warp_r[threadIdx.x >> 5];
		Bx[offset + threadIdx.x * 4 + 2] = partial_b[2] + warp_r[threadIdx.x >> 5];
		Bx[offset + threadIdx.x * 4 + 2] = partial_b[3] + warp_r[threadIdx.x >> 5];

		if (threadIdx.x & 31 == 31 && threadIdx.x >> 5 == 31) {
			chunkAccGate = partial_a[3];
			chunkAccToken = partial_b[3];
		}
	}
}
