
import torch

from scans import full_scan_cuda as full_cuda
from scans import partial_scan_cuda as partial_cuda

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


def plot_vs(x, y1_mean, y2_mean):
	plt.plot(x, y1_mean, linestyle='-', label='contraction scan')
	plt.plot(x, y2_mean, linestyle='-', label='full scan')
	plt.legend()
	plt.title('full scan versus contraction scan')
	plt.xlabel('seqlen')
	plt.ylabel('time')
	plt.savefig(f'vg_seqlen.png')
	plt.clf()


if __name__ == "__main__":

	B, T, D = 1, 256, 2
	d_in = 2
	steps = 100
	vanilla = False
	g_params = {64: 1, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	slen = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
	# dlen =  [128*x for x in range(1, 17)]
	dlen =  [2]

	results = PrettyTable()
	results.field_names = ['B', 'L', 'D', 'full scan cuda (ms)', 'contraction scan cuda (ms)', 'speedup']

	lmean1 = []
	lmean2 = []
	speedups1 = []
	for l, L in enumerate(slen):
		G = L//32 if L not in g_params else g_params[L]
		pg = L // G
		partial_cuda.PScan.idxs = [(x*pg) for x in range(G)]

		tl_mean1 = torch.empty(len(dlen))
		tl_mean2 = torch.empty(len(dlen))
		for d, D in enumerate(dlen):
			timing1 = torch.empty(10)
			timing2 = torch.empty(10)
			timing3 = torch.empty(10)

			for r in range(timing1.size(0)):
				A = torch.rand(B, L, D * d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D * d_in).to('cuda').requires_grad_()
				# warmup
				full_cuda.pscan(A.mT, X.mT)
				partial_cuda.pscan(A.mT, X.mT)

				A = torch.rand(B, L, D * d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D * d_in).to('cuda').requires_grad_()

				# Evaluating the contraction method
				start_time = time.perf_counter()
				for _ in range(steps):
					partial_cuda.pscan(A.mT, X.mT)
				timing1[r] = time.perf_counter() - start_time

				A = torch.rand(B, L, D * d_in).to('cuda').requires_grad_()
				X = torch.randn(B, L, D * d_in).to('cuda').requires_grad_()

				# Evaluating the simple method
				start_time = time.perf_counter()
				for _ in range(steps):
					full_cuda.pscan(A.mT, X.mT)
				timing2[r] = time.perf_counter() - start_time


			t1 = timing1.mean().item() # contraction scan
			t2 = timing2.mean().item() # full scan
			speedup1 = t2 / t1
			results.add_row([B, L, D, t2, t1, speedup1])
			print(f"B={B}, L={L}, D={D}, speedup={speedup1}")

			speedups1.append(speedup1)
			tl_mean1[d] = t1
			tl_mean2[d] = t2
		lmean1.append(tl_mean1.mean().item())
		lmean2.append(tl_mean2.mean().item())

	results.float_format = '0.4'
	print(results)
	results = PrettyTable()
	results.field_names = ['full scan cuda mean (ms)', 'contraction scan cuda mean (ms)', 'speedup']

	overall_t1 = sum(lmean1) / len(lmean1)
	overall_t2 = sum(lmean2) / len(lmean2)
	s1 = sum(speedups1) / len(speedups1)
	results.add_row([overall_t2, overall_t1, s1])
	
	results.float_format = '0.4'
	print(results)

	plot_vs(slen, lmean1, lmean2)
