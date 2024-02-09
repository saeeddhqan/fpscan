

import torch
import random, math, numpy

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)


class PScan(torch.autograd.Function):

	@staticmethod
	def expand_(A, X):
		# Unrolling gains ~8% speed
		if A.size(1) > 4:
			T = 2 * (A.size(1) // 2)
			D, d_in = A.size(-2), A.size(-1)
			Aa = A[:, :T].view(A.size(0), T // 2, 2, D, d_in)
			Xa = X[:, :T].view(X.size(0), T // 2, 2, D, d_in)
			Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
			Aa[:, :, 1].mul_(Aa[:, :, 0])
			PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
			Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
			Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
			if T < A.size(1):
				X[:, -1].add_(A[:, -1].mul(X[:, -2]))
				A[:, -1].mul_(A[:, -2])
		elif A.size(1) == 2:
			X[:, 1].add_(A[:, 1].mul(X[:, 0]))
			A[:, 1].mul_(A[:, 0])
		elif A.size(1) == 3:
			X[:, 1].add_(A[:, 1].mul(X[:, 0]))
			A[:, 1].mul_(A[:, 0])
			X[:, 2].add_(A[:, 2].mul(X[:, 1]))
			A[:, 2].mul_(A[:, 1])
		elif A.size(1) == 4:
			X[:, 1].add_(A[:, 1].mul(X[:, 0]))
			A[:, 1].mul_(A[:, 0])
			X[:, 2].add_(A[:, 2].mul(X[:, 1]))
			A[:, 2].mul_(A[:, 1])
			X[:, 3].add_(A[:, 3].mul(X[:, 2]))
			A[:, 3].mul_(A[:, 2])


	@staticmethod
	def acc_rev_(A, X):
		if A.size(1) > 4:
			T = 2 * (X.size(1) // 2)
			D, d_in = A.size(-2), A.size(-1)
			Aa = A[:, -T:].view(A.size(0), T // 2, 2, D, d_in)
			Xa = X[:, -T:].view(X.size(0), T // 2, 2, D, d_in)
			Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
			B = Aa[:, :, 0].clone()
			B[:, 1:].mul_(Aa[:, :-1, 1])
			PScan.acc_rev_(B, Xa[:, :, 0])
			Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
			if T < A.size(1):
				X[:, 0].add_(A[:, 1].mul(X[:, 1]))
		elif A.size(1) == 2:
			X[:, 0].add_(A[:, 1].mul(X[:, 1]))
		elif A.size(1) == 3:
			X[:, 1].add_(A[:, 2].mul(X[:, 2]))
			X[:, 0].add_(A[:, 1].mul(X[:, 1]))
		elif A.size(1) == 4:
			X[:, 2].add_(A[:, 3].mul(X[:, 3]))
			X[:, 1].add_(A[:, 2].mul(X[:, 2]))
			X[:, 0].add_(A[:, 1].mul(X[:, 1]))


	@staticmethod
	def forward(ctx, A, X, Y_init):
		ctx.A = A.clone()
		ctx.Y_init = Y_init[:, None].clone()
		ctx.A_star = ctx.A.clone()
		ctx.X_star = X.clone()
		PScan.expand_(ctx.A_star, ctx.X_star)
		return ctx.A_star * ctx.Y_init + ctx.X_star


	@staticmethod
	def backward(ctx, grad_output):
		U = grad_output * ctx.A_star
		print(ctx.A_star.shape)
		exit()
		A = ctx.A.clone()
		R = grad_output.clone()
		PScan.acc_rev_(A, R)
		Q = ctx.Y_init.expand_as(ctx.X_star).clone()
		Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
		return (Q * R), R, U.sum(dim=1)


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
	Ax = torch.randn(B, T, D, d_in).to('cuda').requires_grad_()
	Bx = torch.randn(B, T, D, d_in).to('cuda').requires_grad_()
	Y_init = torch.zeros(B, D, d_in).to('cuda').requires_grad_()
	ref = naive_pscan(Ax, Bx, Y_init)
	target = torch.randn_like(ref).to('cuda')

	error = loss_function(ref, target)
	error.backward()
	ref_Ax_gradient = Ax.grad
	ref_Bx_gradient = Bx.grad
	ref_Y_init_gradient = Y_init.grad
	Ax.grad = None
	Bx.grad = None
	Y_init.grad = None

	parallel = pscan(Ax, Bx, Y_init)

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
