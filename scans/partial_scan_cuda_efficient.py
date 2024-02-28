
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline
import random, math, numpy

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)
cuda_source1 = (Path(__file__).parent / 'csrc' / 'shared.h').read_text()
cuda_source2 = (Path(__file__).parent / 'csrc' / 'partial_scan_small_1t.cu').read_text()
cuda_source3 = (Path(__file__).parent / 'csrc' / 'partial_scan_small_2t.cu').read_text()
cuda_source4 = (Path(__file__).parent / 'csrc' / 'partial_scan_small_4t.cu').read_text()
cuda_sourcen = (Path(__file__).parent / 'csrc' / 'partial_scan_mix.cu').read_text()

cpp_source = (Path(__file__).parent / 'csrc' / 'main.cpp').read_text()


module = load_inline(
	name='partial_scan',
	cpp_sources=[cpp_source],
	cuda_sources=[cuda_source1 + cuda_source2 + cuda_source3 + cuda_source4 + cuda_sourcen],
	functions=['partial_scan_forward'],
	verbose=False,
	extra_cuda_cflags=[
		"-O3",
		"-std=c++17",
		"--ptxas-options=-v",
		"-lineinfo",
		"--fmad", "false",
		"-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
		"-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
		'--use_fast_math',
	]
)

scan_forward = module.partial_scan_forward


class PScan(torch.autograd.Function):
	idxs = None

	@staticmethod
	def forward(ctx, A, X, H, Wq, Wk):
		# X = X.contiguous()
		# A = A.contiguous()
		# H = H.contiguous()
		# Wq = Wq.contiguous()
		# Wk = Wk.contiguous()

		# X = scan_forward(A, X)
		X = scan_forward(A, X, H, Wq, Wk)
		# scan_forward(A, X, H, Wq, Wk)
		ctx.save_for_backward(A, X)
		return X

	@staticmethod
	def backward(ctx, grad_output):
		A, X = ctx.saved_tensors
		grad_output = grad_output.contiguous()
		padded_shifted_A = torch.cat([A, torch.ones_like(A[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
		d_X = scan_backward(padded_shifted_A, grad_output)
		padded_outputs = torch.cat([torch.zeros_like(X[:, :, :1]), X], dim=-1)[:, :, :-1]
		padded_outputs[:, :, PScan.idxs] = 0
		return padded_outputs * d_X, d_X, None, None, None


pscan = PScan.apply


def test_correctness(x, y, atol=1e-1):
	assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'


# def naive_pscan(A, X, Bh, dBh, Wq, G):
# 	B, T, C = A.shape
# 	offset = T // G
# 	seqs = []
# 	blocks = []
# 	for g in range(G):
# 		y = torch.zeros_like(Bh[:, g])
# 		o = []
# 		for k in range(offset):
# 			y = A[:, (offset * g) + k] * y + X[:, (offset * g) + k]
# 			o.append(y.unsqueeze(1))
# 		seqs.append(torch.stack(o, dim=1))
# 		blocks.append(y)
	
# 	seqs = torch.stack(seqs, dim=1).view(B, T, C).contiguous()
# 	blocks = torch.stack(blocks, dim=1)
# 	blocks_q = Wq * blocks + blocks
# 	blocks_res = torch.zeros_like(blocks)
# 	for i in range(1, G-1):
# 		tscore = (blocks_q[:, i, None] * dBh[:,[j for j in range(i)]]).mT
# 		for j in range(C):
# 			select2 = torch.max(tscore[:,j], dim=-1)
# 			selected = blocks[torch.arange(B), select2.indices, j]
# 			blocks_res[:, i, j] = selected * select2.values # V * (Q * K)

# 	blocks += blocks_res
# 	for i in range(1, G):
# 		seqs[:, (i * offset): (i*offset)+offset] += blocks[:, i-1, None]
# 	return seqs


def naive_pscan(A, X, Y_init, G):
	offset = A.size(1) // G
	groups = []
	for g in range(G):
		y = Y_init
		o = []
		for k in range(offset):
			y = A[:, (offset * g) + k] * y + X[:, (offset * g) + k]
			o.append(y.unsqueeze(1))
		groups.append(torch.stack(o, dim=1))
	groups = torch.stack(groups, dim=1).view(A.size(0), A.size(1), A.size(2)).contiguous()
	return groups

def loss_function(o, target):
	return torch.sum((o - target) ** 2)


if __name__ == "__main__":
	g_params = {64: 1, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	B, T, D, d_in = 2, 32, 2, 1
	G = T // 32 if T not in g_params else g_params[T]
	pg = T // G
	# Ax = 0.99 + 0.01 * torch.rand(B, T, D * d_in).to('cuda')
	Ax = torch.ones(B, T, D * d_in).to('cuda')
	Bx = torch.ones(B, T, D * d_in).to('cuda')
	Bh = torch.ones(B, G, D * d_in).to('cuda')
	Wq = torch.ones(1, 1, D * d_in).to('cuda')
	Wk = torch.ones(1, 1, D * d_in).to('cuda')
	Y_init = torch.zeros(B, D * d_in).to('cuda')

	ref = naive_pscan(Ax, Bx, Y_init, G)
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

	print(Bx[0])
	parallel = pscan(Ax.mT.contiguous(), Bx.mT.contiguous(), Bh.mT.contiguous(), Wq, Wk).mT.contiguous()
	print(ref[0])
	print(parallel[0])
	# error = loss_function(parallel, target)
	# error.retain_grad()
	# error.backward()

	test_correctness(ref, parallel, atol=1e-3)
	print('passed: similar output')
	# test_correctness(ref_Ax_gradient, Ax.grad, atol=1e-3)
	# print('passed: similar Ax gradient')
	# test_correctness(ref_Bx_gradient, Bx.grad, atol=1e-3)
	# print('passed: similar Bx gradient')
	# print('All passed')
