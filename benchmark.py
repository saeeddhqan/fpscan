
import torch

from scans import full_scan
from scans import contraction_scan

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

	B, T, D = 4, 256, 2
	d_in = 1
	steps = 150
	vanilla = False
	# g_params = {64: 1, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	g_params = {64: 2, 128: 4, 256: 8, 512: 16, 1024: 16, 2048: 32, 4096: 32, 8192: 64, 16384: 128, 32768: 256, 65536: 512}
	slen = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
	# slen = [32, 64, 128, 256, 512, 1024, 2048, 4096]
	# slen = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
	# slen = [4096]
	# dlen =  [128*x for x in range(1, 17)]
	dlen =  [8]

	results = PrettyTable()
	results.field_names = ['B', 'L', 'D', 'full scan cuda (ms)', 'contraction scan cuda (ms)', 'speedup']

	lmean1 = []
	lmean2 = []
	speedups1 = []
	for l, L in enumerate(slen):
		G = g_params[L]
		pg = L // G
		# contraction_scan.PScan.idxs = [(x*pg) for x in range(G)]

		tl_mean1 = torch.empty(len(dlen))
		tl_mean2 = torch.empty(len(dlen))
		for d, D in enumerate(dlen):
			timing1 = torch.empty(10)
			timing2 = torch.empty(10)
			timing3 = torch.empty(10)

			for r in range(timing1.size(0)):
				A = 0.99 + 0.01 * torch.rand(B, L, D * d_in).mT.contiguous().to('cuda')
				# A = torch.ones(B, L, D * d_in).mT.contiguous().to('cuda')
				X = torch.randn(B, L, D * d_in).mT.contiguous().to('cuda')
				Wq = torch.randn(1, 1, D * d_in).mT.contiguous().to('cuda')
				Wk = torch.randn(1, 1, D * d_in).mT.contiguous().to('cuda')
				Bh = torch.randn(B, G, D * d_in).mT.contiguous().to('cuda')
				# warmup
				full_scan.pscan(A, X)
				contraction_scan.pscan(A, X, Bh, Wq, Wk)

				# Evaluating the simple method
				start_time = time.perf_counter()
				for _ in range(steps):
					full_scan.pscan(A, X)
				timing2[r] = time.perf_counter() - start_time
	
				# Evaluating the contraction method
				start_time = time.perf_counter()
				for _ in range(steps):
					contraction_scan.pscan(A, X, Bh, Wq, Wk)
				timing1[r] = time.perf_counter() - start_time


			t1 = timing1.mean().item() # contraction scan
			t2 = timing2.mean().item() # full scan
			speedup1 = t2 / t1
			results.add_row([B, L, D*d_in, t2, t1, speedup1])
			print(f"B={B}, L={L}, D={D * d_in}, speedup={speedup1}")

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
