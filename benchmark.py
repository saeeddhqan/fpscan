
import torch

from scans import pscan1
from scans import pscan2
from scans import pscan_cuda as pscan3

import random, math, numpy, time, sys
import matplotlib.pyplot as plt
from prettytable import PrettyTable

def set_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

set_seed(1244)

def test_correctness(x, y, atol=1e-1):
	assert torch.allclose(x, y, atol=atol), f"Expected {x} to equal {y}"


def naive_pscan(A, X, Y_init):
	y = Y_init
	s = 0
	o = []
	for k in range(A.size(1)):
		y = A[:, k] * y + X[:, k]
		s = s + y
		o.append(y)
	o = torch.stack(o, dim=1)
	return o


def plot_vs(x, y1_mean, y2_mean, y3_mean):
	plt.plot(x, y1_mean, linestyle='-', label='contraction scan')
	plt.plot(x, y2_mean, linestyle='-', label='full scan')
	plt.plot(x, y3_mean, linestyle='-', label='contraction(cuda) scan')
	plt.legend()
	plt.title('full scan versus contraction scan')
	plt.xlabel('seqlen')
	plt.ylabel('time')
	plt.savefig(f'vg_seqlen.png')
	plt.clf()


if __name__ == "__main__":

	B, G, T, D = 2, 8, 256, 32
	d_in = 16
	steps = 100
	vanilla = False
	# slen = slen = [2**x for x in range(4, 10)] + [512*x for x in range(2, 22)] + [1024*x for x in range(11, 16)] + [4096*x for x in range(4, 9)]
	slen = slen = [64, 128, 256, 512, 1024]
	# slen_g = [int(x/16) for x in slen[:-5]] + [x/32 for x in slen[-5:]]
	slen_g = [int(x/16) for x in slen]
	# dlen =  [128*x for x in range(1, 17)]
	dlen =  [4]

	results = PrettyTable()
	results.field_names = ['B', 'L', 'D', 'G', 'full scan (ms)', 'contraction (python) (ms)', 'contraction (cuda) (ms)', 'speedup(python)', 'speedup(cuda)']

	mean1 = []
	mean2 = []
	mean3 = []
	lmean1 = []
	lmean2 = []
	lmean3 = []
	speedups1 = []
	speedups2 = []
	for l, L in enumerate(slen):
		G = int(slen_g[l])
		tl_mean1 = torch.empty(len(dlen))
		tl_mean2 = torch.empty(len(dlen))
		tl_mean3 = torch.empty(len(dlen))
		for d, D in enumerate(dlen):
			timing1 = torch.empty(10)
			timing2 = torch.empty(10)
			timing3 = torch.empty(10)

			for r in range(timing1.size(0)):
				A = 0.9 + 0.1 * torch.rand(B, L, D, d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D, d_in).to('cuda').requires_grad_()
				Y_init1 = torch.randn((B, G, D, d_in)).to('cuda').requires_grad_()
				Y_init2 = torch.randn((B, D, d_in)).to('cuda').requires_grad_()

				# warmup
				Y1 = pscan1.pscan(A, X, Y_init1, G)
				Y2 = pscan2.pscan(A, X, Y_init2)
				Y3 = pscan3.pscan(A, X, Y_init1, G)
				Y_init = torch.fft.fft2(Y1[:, :, -1].float(), dim=(-1, -2)).real


				A = 0.9 + 0.1 * torch.rand(B, L, D, d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D, d_in).to('cuda').requires_grad_()
				Y_init1 = torch.randn((B, G, D, d_in)).to('cuda').requires_grad_()

				# Evaluating the contraction method
				start_time = time.perf_counter()
				for _ in range(steps):
					Y = pscan1.pscan(A, X, Y_init1, G)
					Y_init1 = torch.fft.fft(torch.fft.fft(Y[:, :, -1], dim=-1), dim=-2).real
				timing1[r] = time.perf_counter() - start_time

				A = 0.9 + 0.1 * torch.rand(B, L, D, d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D, d_in).to('cuda').requires_grad_()
				Y_init2 = torch.randn((B, D, d_in)).to('cuda').requires_grad_()

				# Evaluating the simple method
				start_time = time.perf_counter()
				for _ in range(steps):
					Y = pscan2.pscan(A, X, Y_init2)
				timing2[r] = time.perf_counter() - start_time

				A = 0.9 + 0.1 * torch.rand(B, L, D, d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D, d_in).to('cuda').requires_grad_()
				Y_init1 = torch.randn((B, G, D, d_in)).to('cuda').requires_grad_()

				# Evaluating the contraction (cuda) method
				start_time = time.perf_counter()
				for _ in range(steps):
					Y = pscan3.pscan(A, X, Y_init1, G)
					Y_init1 = torch.fft.fft(torch.fft.fft(Y[:, :, -1], dim=-1), dim=-2).real
				timing3[r] = time.perf_counter() - start_time

			t1 = timing1.mean().item() # contraction scan
			t2 = timing2.mean().item() # full scan
			t3 = timing3.mean().item() # contraction (cuda) scan
			speedup1 = t2 / t1
			speedup2 = t2 / t3
			results.add_row([B, L, D, G, t2, t1, t3, speedup1, speedup2], divider=(True if l == len(slen)-1 else False))
			print(f"B={B}, G={G}, L={L}, D={D}, speedup(python)={speedup1}, speedup(cuda)={speedup2}")

			speedups1.append(speedup1)
			speedups2.append(speedup2)
			mean1.append(t1)
			mean2.append(t2)
			mean3.append(t3)
			tl_mean1[d] = t1
			tl_mean2[d] = t2
			tl_mean3[d] = t3
		lmean1.append(tl_mean1.mean().item())
		lmean2.append(tl_mean2.mean().item())
		lmean3.append(tl_mean3.mean().item())

	overall_t1 = sum(lmean1) / len(lmean1)
	overall_t2 = sum(lmean2) / len(lmean1)
	overall_t3 = sum(lmean3) / len(lmean1)
	s1 = sum(speedups1) / len(speedups1)
	s2 = sum(speedups2) / len(speedups2)
	results.add_row(['-', '-', '-', '-', overall_t2, overall_t1, overall_t3, s1, s2])
	results.float_format = "0.2"
	print(results)
	plot_vs(slen, lmean1, lmean2, lmean3)
