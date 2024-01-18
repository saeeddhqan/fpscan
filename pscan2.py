
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
		if A.size(2) > 4:
			B, G = A.shape[:2]
			T = 2 * (A.size(2) // 2)
			Aa = A[:, :, :T].view(B, G, T // 2, 2, -1, 1)
			Xa = X[:, :, :T].view(B, G, T // 2, 2, -1, X.size(-1))
			Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
			Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
			PScan.expand_(Aa[:, :, :, 1], Xa[:, :, :, 1])
			Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
			Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])
			if T < A.size(2):
				X[:, :, -1].add_(A[:, :, -1].mul(X[:, :, -2]))
				A[:, :, -1].mul_(A[:, :, -2])
		elif A.size(2) == 2:
			X[:, 1].add_(A[:, :, 1].mul(X[:, :, 0]))
			A[:, 1].mul_(A[:, :, 0])
		elif A.size(2) == 3:
			X[:, 1].add_(A[:, :, 1].mul(X[:, :, 0]))
			A[:, 1].mul_(A[:, :, 0])
			X[:, 2].add_(A[:, :, 2].mul(X[:, :, 1]))
			A[:, 2].mul_(A[:, :, 1])
		elif A.size(2) == 4:
			X[:, :, 1].add_(A[:, :, 1].mul(X[:, :, 0]))
			A[:, :, 1].mul_(A[:, :, 0])
			X[:, :, 2].add_(A[:, :, 2].mul(X[:, :, 1]))
			A[:, :, 2].mul_(A[:, :, 1])
			X[:, :, 3].add_(A[:, :, 3].mul(X[:, :, 2]))
			A[:, :, 3].mul_(A[:, :, 2])

	@staticmethod
	def acc_rev_(A, X):
		if A.size(2) > 4:
			B, G = A.shape[:2]
			T = 2 * (X.size(2) // 2)
			Aa = A[:, :, -T:].view(B, G, T // 2, 2, -1, 1)
			Xa = X[:, :, -T:].view(B, G, T // 2, 2, -1, X.size(-1))
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
		B, T = A.shape[:2]
		D = X.size(-1)
		ctx.A = A.view(B, G, T // G).unsqueeze(-1).clone()
		ctx.Y_init = Y_init.unsqueeze(2).clone()
		ctx.A_star = ctx.A.clone()
		ctx.X_star = X.view(B, G, T // G, D).clone()

		PScan.expand_(ctx.A_star, ctx.X_star)
		return ctx.A_star * ctx.Y_init + ctx.X_star

	@staticmethod
	def backward(ctx, grad_output):
		U = grad_output * ctx.A_star
		A = ctx.A.clone()
		R = grad_output.clone()
		PScan.acc_rev_(A, R)
		Q = ctx.Y_init.expand_as(ctx.X_star).clone()
		Q[:, :, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :, :-1])
		return (Q * R).sum(-1), R, U.sum(dim=2)


pscan = PScan.apply
