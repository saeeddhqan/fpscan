'''
Let me explain it. Suppose the input values are like the following,
and the number of input tokens is 16:

1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,

and the number of groups is 4. In this case, we divide the input
tokens into 4 groups:

1, 1, 1, 1    2, 2, 2, 2    3, 3, 3, 3    4, 4, 4, 4

a simple scan on each group, makes them (somehow) look like this:

1, 2, 3, 4    2, 4, 6, 8    3, 6, 9, 12    4, 8, 12, 16

So far, the tokens in group four, are not aware of group three, two, and so on.
Therefore, we require another communications that happens on the final
tokens of each group:

4, 8, 12, 16

Now, we use a different communication algorithm that works like an attention:
Each value has a q, k, v.
To update value of group four (16):

scores = 16thq * [4th k, 8th k, 12th k] =
	[16th q * 4th k, 16th q * 8th k, 16th q * 12th k]

selected_score = max(scores) =
	max([16th q * 4th k, 16th q * 8th k, 16th q * 12th k]) = 16th q * 12th k

16th v += selected_score * 12th v

And we do the same for the third group, second group, etc.

This method works on each dimension of each token. In this way, each dimension
can select one token to communicate with.

We create q, k, and v in the following way:

q = delta_q * v + v
k = delta_k * Bh + Bh # Bh is the hidden states of each group from the previous layer.
v = The final value of each group. i.e here 4 is the v for the first group and so on.

delta: these are trainable paramaters

-----

Ax, Bx: In this code Ax, and Bx are for scan.
Bh: the hidden states of each group, from the previous layer
Wq, Wk: deltas for adjusting tensors to make q, and k

g_params: sequence lengths with their corresponding groups number
'''



from pathlib import Path

import torch
import math
from something_weird import dimwise_forward
from einops import rearrange
import random, math, numpy

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)


def test_correctness(x, y, atol=1e-1):
	assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'


class scanFunc(torch.autograd.Function):
	idxs = None
	@staticmethod
	def forward(ctx, Ax, Bx, Bh, Wq, Wk):
		dimwise_forward(Ax, Bx, Bh, Wq, Wk)
		ctx.save_for_backward(Ax, Bx)
		return Bx

	@staticmethod
	def backward(ctx, dout):
		input, weight, bias = ctx.saved_tensors
		dout  = dout.contiguous()
		du, dk, dbias = dimwise_backward(dout, input, weight, bias, ctx.padding, ctx.is_bhl)
		return du, dk, dbias, None, None


pscan = scanFunc.apply

def contraction_pscan(Ax, Bx, Bh, kBh, Wq, G, chunklen=2):
	B, T, C = Ax.shape
	offset = T // G
	cseqs = []
	chunk_b = None
	for i in range(chunklen):
		pG = i * (G//chunklen)
		eG = pG + (G//chunklen)

		seqs = []
		blocks = []
		for g in range(pG, eG):
			y = torch.zeros_like(Bh[:, g])
			o = []
			for k in range(offset):
				if i > 0 and k == 0 and g == pG:
					y = Ax[:, (offset * g) + k] * chunk_b + Bx[:, (offset * g) + k]
				else:
					y = Ax[:, (offset * g) + k] * y + Bx[:, (offset * g) + k]
				o.append(y.unsqueeze(1))
			seqs.append(torch.stack(o, dim=1))
			blocks.append(y)
		
		seqs = torch.stack(seqs, dim=1).view(B, T // chunklen, C).contiguous()
		blocks = torch.stack(blocks, dim=1)

		blocks_q = Wq * blocks + blocks
		# print(kBh.shape)
		# print(kBh[0])
		blocks_res = torch.zeros_like(blocks)
		for x in range(1, (G // chunklen) - 1):
			tscore = (blocks_q[:, x, None] * kBh[:,[t for t in range(x)]]).mT # Q * K
			for j in range(C):
				select2 = torch.max(tscore[:,j], dim=-1)
				selected = blocks[torch.arange(B), select2.indices, j]
				blocks_res[:, x, j] = selected * select2.values # Vp * (Q * K)

		blocks += blocks_res # Vt + (Vp * Q * K)
		for x in range(1, G // chunklen):
			seqs[:, (x * offset): (x*offset)+offset] += blocks[:, x-1, None]
		
		cseqs.append(seqs)

		chunk_b = seqs[:, -1]
	cseqs = torch.stack(cseqs, dim=1).view(B, T, C).contiguous()
	return cseqs

# def naive_pscan(A, X, Y_init, G):
# 	offset = A.size(1) // G
# 	groups = []
# 	for g in range(G):
# 		y = Y_init
# 		o = []
# 		for k in range(offset):
# 			y = A[:, (offset * g) + k] * y + X[:, (offset * g) + k]
# 			o.append(y.unsqueeze(1))
# 		groups.append(torch.stack(o, dim=1))
# 	groups = torch.stack(groups, dim=1).view(A.size(0), A.size(1), A.size(2)).contiguous()
# 	return groups


if __name__ == "__main__":
	g_params = {64: 2, 128: 4, 256: 8, 512: 16, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	c_params = {8192: 2, 16384: 4, 32768: 8, 65536: 16}
	B, T, D, d_in = 1, 65536, 2, 1
	# print(list(g_params.keys())[-2:])
	for T in list(g_params.keys()):
		T = 4096
		G = g_params[T]
		C = 1 if T <= 4096 else c_params[T]
		print(f'benchmarking {T}t, {G}g, {C}c')
		pg = T // G
		Ax = 0.99 + 0.01 * torch.rand(B, T, D * d_in).to('cuda')
		# Ax = torch.ones(B, T, D * d_in).to('cuda')
		Bx = torch.randn(B, T, D * d_in).to('cuda')
		Bh = torch.randn(B, G, D * d_in).to('cuda')
		Wq = torch.randn(1, 1, D * d_in).to('cuda')
		Wk = torch.randn(1, 1, D * d_in).to('cuda')
		# ref = contraction_pscan(Ax, Bx, Bh, (Bh * Wk) + Bh, Wq, G, C)
		# print(ref)
		# target = torch.randn_like(ref).to('cuda')

		# error = loss_function(ref, target)
		# error.retain_grad()
		# error.backward()
		# ref_Ax_gradient = Ax.grad
		# ref_Bx_gradient = Bx.grad
		# ref_Y_init_gradient = Y_init.grad
		# Ax.grad = None
		# Bx.grad = None
		# Y_init.grad = None


		# PScan.idxs = [(x*pg) for x in range(G)]
		# print(ref[:, 4220:4226].tolist())
		for _ in range(100):
			parallel = pscan(Ax.mT.contiguous(), Bx.mT.contiguous(), Bh.mT.contiguous(), Wq, Wk).mT.contiguous()
		# error = loss_function(parallel, target)
		# error.retain_grad()
		# error.backward()

		# test_correctness(ref[:, 4225:4226], parallel[:, 4225:4226], atol=1e-3)
		# test_correctness(ref, parallel, atol=1e-2)
		# print(f'[{T}]passed: similar output')
