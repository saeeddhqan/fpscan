
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

cuda_source = (Path(__file__).parent / 'csrc' / 'full_scan' / 'full_scan.cu').read_text()

cpp_source = """
at::Tensor warpscan_forward(const at::Tensor &gates, const at::Tensor &tokens, const at::Tensor &out, const bool reverse);
"""

module = load_inline(
	name='warpscan',
	cpp_sources=[cpp_source],
	cuda_sources=[cuda_source],
	functions=['warpscan_forward'],
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
warpscan_forward = module.warpscan_forward

def scan_forward(gates, tokens, reverse=False):
	output = torch.zeros_like(tokens)
	warpscan_forward(gates, tokens, output, reverse)
	return output


class PScan(torch.autograd.Function):
	@staticmethod
	def forward(ctx, A, X):
		X = scan_forward(A, X)
		ctx.save_for_backward(A, X)
		return X

	@staticmethod
	def backward(ctx, grad_output):
		A, X = ctx.saved_tensors
		grad_output = grad_output.contiguous()

		padded_shifted_A = torch.cat([A, torch.ones_like(A[:, :, :1])], dim=-1)[:, :, 1:].contiguous()
		d_X = scan_forward(padded_shifted_A, grad_output, reverse=True)
		padded_outputs = torch.cat([torch.zeros_like(X[:, :, :1]), X], dim=-1)[:, :, :-1]
		return padded_outputs * d_X, d_X


pscan = PScan.apply

if __name__ == "__main__":
	def test_correctness(x, y, atol=1e-1):
		assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'

	def naive_pscan(A, X, Y_init):
		y = Y_init
		o = []
		for k in range(A.size(1)):
			y = A[:, k] * y + X[:, k]
			o.append(y)
		o = torch.stack(o, dim=1)
		return o

	def loss_function(o, target):
		return torch.sum((o - target) ** 2)

	B, T, D, d_in = 4, 32, 2, 2
	Ax = torch.ones(B, T, D * d_in).to('cuda').requires_grad_()
	Bx = torch.ones(B, T, D * d_in).to('cuda').requires_grad_()
	Y_init = torch.zeros(B, D * d_in).to('cuda').requires_grad_()
	ref = naive_pscan(Ax, Bx, Y_init)
	target = torch.randn_like(ref).to('cuda')

	error = loss_function(ref, target)
	error.backward()
	ref_Ax_gradient = Ax.grad
	ref_Bx_gradient = Bx.grad
	Ax.grad = None
	Bx.grad = None

	parallel = pscan(Ax.mT, Bx.mT).mT.contiguous()
	print(parallel[0])
	error = loss_function(parallel, target.view(B, T, -1))
	error.retain_grad()
	error.backward()

	print(ref_Ax_gradient[0])
	print(Ax.grad[0])

	test_correctness(ref, parallel, atol=1e-1)
	print('passed: similar output')
	test_correctness(ref_Ax_gradient, Ax.grad, atol=1e-1)
	print('passed: similar Ax gradient')
	test_correctness(ref_Bx_gradient, Bx.grad, atol=1e-1)
	print('passed: similar Bx gradient')
	print('All passed')
