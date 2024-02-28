
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

cuda_source = (Path(__file__).parent / 'csrc' / 'partial_scan_correct.cu').read_text()
cpp_source = (Path(__file__).parent / 'csrc' / 'main.cpp').read_text()


module = load_inline(
	name='partial_scan',
	cpp_sources=[cpp_source],
	cuda_sources=[cuda_source],
	functions=['partial_scan_forward', 'partial_scan_backward'],
	verbose=True,
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
scan_backward = module.partial_scan_backward


class PScan(torch.autograd.Function):
	idxs = None

	@staticmethod
	def forward(ctx, A, X):
		X = X.contiguous()
		A = A.contiguous()

		X = scan_forward(A, X)
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
		return padded_outputs * d_X, d_X, None

pscan = PScan.apply

if __name__ == "__main__":
	def test_correctness(x, y, atol=1e-1):
		assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'

	def naive_pscan(A, X, Y_init, G):
		offset = A.size(1) // G
		seqs = []
		blocks = []
		for g in range(G):
			y = Y_init
			o = []
			for k in range(offset):
				y = A[:, (offset * g) + k] * y + X[:, (offset * g) + k]
				o.append(y.unsqueeze(1))
			seqs.append(torch.stack(o, dim=1))
			blocks.append(y)
		seqs = torch.stack(seqs, dim=1).view(A.size(0), A.size(1), A.size(2)).contiguous()
		blocks = torch.stack(blocks, dim=1)
		blocks_res = torch.zeros_like(blocks)
		for i in range(1, G):
			tscore = (blocks[:, i, None] * A[:,[j * offset for j in range(i)]]).mT
			for j in range(A.size(-1)):
				select2 = torch.max(tscore[:,j], dim=-1)
				selected = blocks[torch.arange(A.size(0)), select2.indices, j]
				blocks_res[:, i, j] = selected * select2.values

		blocks += blocks_res

		for i in range(1, G):
			seqs[:, i * G: (i*G)+G] += blocks[:, i, None]
		return seqs

	def loss_function(o, target):
		return torch.sum((o - target) ** 2)

	g_params = {64: 1, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	B, T, D, d_in = 2, 32, 2, 2
	G = T // 32 if T not in g_params else g_params[T]
	pg = T // G
	Ax = torch.randn(B, 32, D * d_in).to('cuda')
	Bx = torch.randn(B, 32, D * d_in).to('cuda')
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
	parallel = pscan(Ax.mT, Bx.mT).mT.contiguous()

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
