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

cuda_source = (Path(__file__).parent / 'csrc' / 'myscan.cu').read_text()
cpp_source = (Path(__file__).parent / 'csrc' / 'main.cpp').read_text()

myscan = load_inline(
    name='myscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['myscan_forward'],
    verbose=False,
    extra_cuda_cflags=[
        '-O3',
        '-std=c++17',
        '--ptxas-options=-v',
        '-lineinfo',
        '--fmad', 'false',
        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', 
        '--use_fast_math',
    ]
)
myscan_forward = myscan.myscan_forward


class PScan(torch.autograd.Function):
	@staticmethod
	def acc_rev_(A, X):
		if A.size(2) > 4:
			B, G, T, D, d_in = A.shape
			T = 2 * (X.size(2) // 2)
			Aa = A[:, :, -T:].view(B, G, T // 2, 2, D, d_in)
			Xa = X[:, :, -T:].view(B, G, T // 2, 2, D, d_in)

			Xa[:, :, :, 0].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 1]))
			B = Aa[:, :, :, 0].clone()
			B[:, :, 1:].mul_(Aa[:, :, :-1, 1])
			PScan.acc_rev_(B, Xa[:, :, :, 0])
			Xa[:, :, :-1, 1].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, 1:, 0]))
			if T < A.size(2):
				X[:, :, 0].add_(A[:, :, 1].mul(X[:, :, 1]))
		elif A.size(2) == 2:
			X[:, :, 0].add_(A[:, :, 1].mul(X[:, :, 1]))
		elif A.size(2) == 3:
			X[:, :, 1].add_(A[:, :, 2].mul(X[:, :, 2]))
			X[:, :, 0].add_(A[:, :, 1].mul(X[:, :, 1]))
		elif A.size(2) == 4:
			X[:, :, 2].add_(A[:, :, 3].mul(X[:, :, 3]))
			X[:, :, 1].add_(A[:, :, 2].mul(X[:, :, 2]))
			X[:, :, 0].add_(A[:, :, 1].mul(X[:, :, 1]))

	@staticmethod
	def forward(ctx, A, X, Y_init, G):
		ctx.B, ctx.T, ctx.D, ctx.d_in = A.shape
		ctx.G = G
		ctx.GT = ctx.T // G
		ctx.A = A.clone()
		ctx.Y_init = Y_init[:, :, None].clone()
		ctx.A_star = ctx.A.clone()
		ctx.X_star = X.clone()

		ctx.X_star, ctx.A_star = myscan_forward(
			ctx.A_star,
			ctx.X_star,
			G,
		)
		return (ctx.A_star * ctx.Y_init + ctx.X_star)

	@staticmethod
	def backward(ctx, grad_output):
		U = grad_output * ctx.A_star
		A = ctx.A.view(ctx.B, ctx.G, ctx.GT, ctx.D, ctx.d_in).clone()
		v = (ctx.B, ctx.T, ctx.D, ctx.d_in)
		R = grad_output.clone()
		PScan.acc_rev_(A, R)
		Q = ctx.Y_init.expand_as(ctx.X_star).clone()
		Q[:, :, 1:].mul_(ctx.A_star[:, :, :-1]).add_(ctx.X_star[:, :, :-1])
		return ((Q * R).view(v),
				R.view(v),
				U.sum(dim=2),
				None)


pscan = PScan.apply

if __name__ == "__main__":
	def test_correctness(x, y, atol=1e-1):
		assert torch.allclose(x, y, atol=atol), 'Tensor mismatch'

	def naive_pscan(A, X, Y_init):
		offset = A.size(1) // G
		groups = []

		for g in range(G):
			y = Y_init[:, g]
			o = []
			for k in range(offset):
				y = A[:, (offset * g) + k] * y + X[:, (offset * g) + k]
				o.append(y)
			groups.append(torch.stack(o, dim=1))

		groups = torch.stack(groups, dim=1).view(A.size(0), A.size(1), A.size(2), A.size(3))
		return groups

	def loss_function(o, target):
		return torch.sum((o - target) ** 2)

	B, T, D, d_in = 4, 128, 8, 2
	G = 4
	Ax = torch.randn(B, T, D, d_in).to('cuda').requires_grad_()
	Bx = torch.randn(B, T, D, d_in).to('cuda').requires_grad_()
	Y_init = torch.randn(B, G, D, d_in).to('cuda').requires_grad_()
	ref = naive_pscan(Ax, Bx, Y_init)
	target = torch.randn_like(ref).to('cuda')

	error = loss_function(ref, target)
	error.retain_grad()
	error.backward()
	ref_Ax_gradient = Ax.grad
	ref_Bx_gradient = Bx.grad
	ref_Y_init_gradient = Y_init.grad
	Ax.grad = None
	Bx.grad = None
	Y_init.grad = None

	parallel = pscan(Ax, Bx, Y_init, G).view(B, T, D, d_in)

	error = loss_function(parallel, target)
	error.retain_grad()
	error.backward()

	test_correctness(ref, parallel, atol=1e-1)
	print('passed: similar output')
	test_correctness(ref_Ax_gradient, Ax.grad, atol=1e-1)
	print('passed: similar Ax gradient')
	test_correctness(ref_Bx_gradient, Bx.grad, atol=1e-1)
	print('passed: similar Bx gradient')
	test_correctness(ref_Y_init_gradient, Y_init.grad, atol=1e-1)
	print('passed: similar Y_init gradient')
	print('All passed')
