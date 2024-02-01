import torch
from tqdm.auto import tqdm
from tabulate import tabulate
import csv

from benchmark import benchmark_forward, benchmark_backward, benchmark_memory
import model


def set_B_H(B, H, seqlen):
	if seqlen == 16384 and B > 32:
		B = 32
	if seqlen == 32768 and B > 16:
		B = 16
	if seqlen == 65536 and B > 8:
		B = 8
	if seqlen == 131072 and B > 8:
		B = 8
	if seqlen == 131072 and H > 384:
		H = 384
	if seqlen == 262144 and B > 8:
		B = 8
	if seqlen == 262144 and H > 192:
		H = 192
	if seqlen == 524288 and B > 8:
		B = 8
	if seqlen == 524288 and H > 96:
		H = 96
	if seqlen == 1048576 and B > 8:
		B = 8
	if seqlen == 1048576 and H > 48:
		H = 48
	if seqlen == 2097152 and B > 8:
		B = 8
	if seqlen == 2097152 and H > 32:
		H = 32
	if seqlen == 4194304 and B > 8:
		B = 8
	if seqlen == 4194304 and H > 16:
		H = 16
	return B, H

def set_repeats(seqlen):
	if seqlen <= 4096:
		return 20
	elif seqlen <= 32 * 32768:
		return 10
	else:
		return 5

save_filename = 'benchmark_results.csv'

B = 64
H = 768
total_seqs = B * H
dtype = torch.float32
device = 'cuda'
# seqlens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 32 * 32768, 64 * 32768, 128 * 32768]
seqlens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 32 * 32768]

benchmark_fns = ['forward', 'backward', 'memory']
benchmark_fn_mapping = {
	'forward': benchmark_forward,
	'backward': benchmark_backward,
	'memory': benchmark_memory,
}
funcs = ['mamba', 'contract mamba']
write_tex = True

contraction_vals = {}
ref_vals = {}
savings_vals = {}
keys = []


for benchmark_fn_name in benchmark_fns:
	torch.cuda.empty_cache()
	print(f'Benchmarking {benchmark_fn_name} for {func} with seqlens {seqlens}')

	ref = []
	contraction = []
	savings = []


	for seqlen in tqdm(seqlens):
		N = seqlen

		local_B, local_H = set_B_H(B, H, seqlen)
		repeats = set_repeats(seqlen)
		
		adjustment = total_seqs / (local_B * local_H)

		u = torch.randn(local_B, N, local_H, dtype=dtype).to(device)
		model.config.dim = local_H

		model.config.group = False
		model.config.ngroups = seqlen // 32
		mamba2 = model.Model().to(device)
		benchmark_fn = benchmark_fn_mapping[benchmark_fn_name]

		u.requires_grad = True

		if benchmark_fn_name in ['forward', 'backward']:
			t, m = benchmark_fn(mamba2, u, repeats=repeats, desc=f"Contraction Mamba, {seqlen}", verbose=False)
			contraction.append(m.mean * 1000 * adjustment)
		else:
			m = benchmark_fn(mamba2, u, desc=f"Contraction Mamba, {seqlen}", verbose=False)
			contraction.append(m * adjustment)


	for seqlen in tqdm(seqlens):
		N = seqlen

		local_B, local_H = set_B_H(B, H, seqlen)
		repeats = set_repeats(seqlen)
		
		adjustment = total_seqs / (local_B * local_H)

		u = torch.randn(local_B, N, local_H, dtype=dtype).to(device)
		model.config.dim = local_H

		model.config.group = False
		mamba1 = model.Model().to(device)
		benchmark_fn = benchmark_fn_mapping[benchmark_fn_name]

		u.requires_grad = True

		if benchmark_fn_name in ['forward', 'backward']:
			t, m = benchmark_fn(mamba1, u, repeats=repeats, desc=f"Mamba, {seqlen}", verbose=False)
			ref.append(m.mean * 1000 * adjustment)
		else:
			m = benchmark_fn(mamba1, u, desc=f"Mamba, {seqlen}", verbose=False)
			ref.append(m * adjustment)

		savings = [ r / c for r, c in zip(ref, contraction)]

	print('Contraction', contraction)
	print('Ref', ref)
	print('Savings', savings)
	contraction_vals[(func, benchmark_fn_name)] = contraction
	ref_vals[(func, benchmark_fn_name)] = ref
	savings_vals[(func, benchmark_fn_name)] = savings
	keys.append((func, benchmark_fn_name))

print('Seqlens:', seqlens)
print('Contraction:', contraction_vals)
print('Ref:', ref_vals)
print('Savings:' , savings_vals)

table = [
	['Method'] + seqlens
]

for k in keys:
	table.append([f'{k[0]}, {k[1]}, Contraction Mamba'] + contraction_vals[k])
	table.append([f'{k[0]}, {k[1]}, Mamba'] + ref_vals[k])
	table.append([f'{k[0]}, {k[1]}, Savings'] + savings_vals[k])

print(f'Saving results as {save_filename}')
with open(save_filename, 'w') as f:
	writer = csv.writer(f)
	for row in table:
		writer.writerows(table)

print(tabulate(table))

if write_tex:
	for k in keys:
		header = [['\\textbf{Seq Len}', '\\textbf{PyTorch}', '\\textbf{\\sysname}', '\\textbf{Memory Reduction}' if 'memory' in k[1] else '\\textbf{Speedup}']]
		table_data = header + [
			['\\textbf{' + str(seqlen) + '}', '%1.2f' % ref_vals[k][i], '%1.2f' % contraction_vals[k][i], '%1.2f$\\times$' % savings_vals[k][i]]
			for i, seqlen in enumerate(seqlens)
		]
		latex_table = tabulate(table_data, tablefmt='latex_raw')
		print(f'{k[0]}, {k[1]}, LaTex')
		print(latex_table)
