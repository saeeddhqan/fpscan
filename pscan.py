
import torch
import pscan2
import random, math, numpy
import matplotlib.pyplot as plt

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)

class PScan(torch.autograd.Function):
	# Given A is NxTxMx1 and X is NxTxMxD, expands A and X in
	# place in O(T), and O(log(T)) if not core-bounded, so that
	#
	# Y[:, 0] = Y_init
	# Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
	#
	# can be computed as
	#
	# Y[:, t] = A[:, t] * Y_init + X[:, t]

	@staticmethod
	def expand_(A, X):
		# Unrolling gains ~8% speed

		if A.size(1) > 4:
			T = 2 * (A.size(1) // 2)
			Aa = A[:, :T].view(A.size(0), T // 2, 2, -1, 1)
			Xa = X[:, :T].view(X.size(0), T // 2, 2, -1, X.size(-1))
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
			Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1, 1)
			Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1, X.size(-1))
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

	# A is NxT, X is NxTxD, Y_init is NxD
	#
	# returns Y of same shape as X, with
	#
	# Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
	#         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

	@staticmethod
	def forward(ctx, A, X, Y_init):
		ctx.A = A.unsqueeze(-1).clone()
		ctx.Y_init = Y_init[:, None].clone()
		ctx.A_star = ctx.A.clone()
		ctx.X_star = X.clone()
		PScan.expand_(ctx.A_star, ctx.X_star)
		return ctx.A_star * ctx.Y_init + ctx.X_star

	@staticmethod
	def backward(ctx, grad_output):
		U = grad_output * ctx.A_star
		A = ctx.A.clone()
		R = grad_output.clone()
		PScan.acc_rev_(A, R)
		Q = ctx.Y_init.expand_as(ctx.X_star).clone()
		Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
		return (Q * R).sum(-1), R, U.sum(dim=1)


pscan = PScan.apply


def naive_pscan(A, X, Y_init):
	y = Y_init
	s = 0
	o = []
	for k in range(A.size(1)):
		y = A[:, k, None] * y + X[:, k]
		s = s + y
		o.append(y)
	o = torch.stack(o, dim=1)
	return o

def show(x, y, x_title, y_title, name, title):
	x = numpy.array(x)
	y = numpy.array(y)
	plt.plot(x, y, marker='o', linestyle='-')
	torch.save(y, name + '.pt')
	plt.title(title)
	plt.xlabel(x_title)
	plt.ylabel(y_title)
	plt.savefig(name + '.png')
	# plt.show()

def plot_vs(x):
	for method in ('seqlen', 'dim'):
		y0_mean = torch.load(f'vanilla_{method}_mean.pt')
		y1_mean = torch.load(f'grouped_{method}_mean.pt')
		plt.plot(x[method], y0_mean, linestyle='-', label='vanilla')
		plt.plot(x[method], y1_mean, linestyle='-', label='grouped')
		plt.legend()
		plt.title('vanilla versus grouped')
		plt.xlabel(method)
		plt.ylabel('duration')
		plt.savefig('vg_seqlen.png')
		# plt.show()

if __name__ == "__main__":
	import time, sys

	B, G, T, D = 16, 4, 128, 32
	steps = 100
	vanilla = True
	seq_test = True
	slen = slen = [2**x for x in range(4, 10)] + [512*x for x in range(2, 22)] + [1024*x for x in range(11, 16)] + [4096*x for x in range(4, 9)]
	slen_g = [x/16 for x in slen[:-5]] + [x/32 for x in slen[-5:]]
	dlen = length = [128*x for x in range(1, 17)]

	# plot_vs({'seqlen': slen, 'dim': dlen})
	# exit()

	metric = slen if seq_test else dlen
	slen_mean = []
	slen_std = []
	dlen_mean = []
	dlen_std = []

	for i in metric:
		timing = torch.empty(10)
		T = i if seq_test else T
		D = i if not seq_test else D

		for r in range(timing.size(0)):
			A = 0.9 + 0.1 * torch.rand(B, T, dtype=torch.float64).to('cuda').requires_grad_()
			X = torch.randn(B, T, D, dtype=torch.float64).to('cuda').requires_grad_()
			Y_init = torch.randn((B, D) if vanilla else (B, G, D), dtype=torch.float64).to('cuda').requires_grad_()

			for _ in range(2):
				if vanilla:
					Y1 = pscan(A, X, Y_init)
				else:
					##### second approach
					Y2 = pscan2.pscan(A, X, Y_init, G)
					# We actually don't require the last hstates. But for the sake of performance
					Y_init = torch.fft.fft(torch.fft.fft(Y2[:, :, -1], dim=-1), dim=-2).real


		for r in range(timing.size(0)):
			A = 0.9 + 0.1 * torch.rand(B, T, dtype=torch.float64).to('cuda').requires_grad_()
			X = torch.randn(B, T, D, dtype=torch.float64).to('cuda').requires_grad_()
			Y_init = torch.randn((B, D) if vanilla else (B, G, D), dtype=torch.float64).to('cuda').requires_grad_()

			start_time = time.perf_counter()
			for _ in range(steps):
				if vanilla:
					Y1 = pscan(A, X, Y_init)
				else:
					##### second approach
					Y2 = pscan2.pscan(A, X, Y_init, G)
					# We actually don't require the last hstates. But for the sake of performance
					Y_init = torch.fft.fft(torch.fft.fft(Y2[:, :, -1], dim=-1), dim=-2).real
			duration = time.perf_counter() - start_time
			timing[r] = duration

		print(f"B={B}, G={G}, T={T}, D={D}, duration={timing.mean()} (+/- {timing.std()})")
		if seq_test:
			slen_mean.append(timing.mean())
			slen_std.append(timing.std())
		else:
			dlen_mean.append(timing.mean())
			dlen_std.append(timing.std())

	fname = 'vanilla' if vanilla else 'grouped'
	if seq_test:
		show(slen, slen_mean, 'seq_len', 'duration', fname + '_seqlen_mean', 'seq_len duration mean',)
		show(slen, slen_std, 'seq_len', 'duration', fname + '_seqlen_std', 'seq_len duration std',)
	else:
		show(dlen, dlen_mean, 'dim', 'duration', fname + '_dim_mean', 'dim duration mean',)
		show(dlen, dlen_std, 'dim', 'duration', fname + '_dim_std', 'dim duration std',)


