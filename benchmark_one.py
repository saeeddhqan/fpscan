import torch 
import time
from torch import nn
from einops import rearrange

from prettytable import PrettyTable
import model
#correctness test
def test_correctness(x, y, atol=1e-1):
	assert torch.allclose(x, y, atol=atol), f"Expected {x} to equal {y}"

torch.manual_seed(1234)

dtype = torch.float32
nbytes = 4
device = 'cuda'

torch.set_default_device(device)
torch.set_default_dtype(dtype)
   

repeats = 100


results = PrettyTable()
results.field_names = ['B', 'L', 'D', 'torch time (ms)', 'cuda time (ms)', 'speedup', 'Effective bandwidth (GB/s)', 'TFLOPS']

for b in [16]:
	for l in [1024, 2048, 4096, 8192]:
		for d in [768, 1024, 2048, 8192]:
			x = torch.randn([b, l, d])
			model.config.dim = d

			model.config.group = False
			vanilla_mamba = model.Model()

			model.config.group = True
			model.config.ngroups = l // 32
			contract_mamba = model.Model()

			#warmup
			y_torch = vanilla_mamba(x)
			
			torch.cuda.synchronize()
			start = time.time()
			for _ in range(repeats):
				y_torch = vanilla_mamba(x)
			torch.cuda.synchronize()
			torch_time = (time.time() - start)*1000/repeats

			# warmup
			y_cuda = contract_mamba(x)

			torch.cuda.synchronize()
			start = time.time()
			for _ in range(repeats):
				y_cuda = contract_mamba(x)
			torch.cuda.synchronize()
			cuda_time = (time.time() - start)*1000/repeats

			speedup = torch_time / cuda_time

			effective_bandwidth = (b * l * d * 2 * nbytes + d * nbytes) / (cuda_time * 1e-3) / (2**30)

			l_out = l # change
			tera_flops = (b * l_out * d * 2) / (cuda_time * 1e-3) / (2**40)
			results.add_row([b, l, d, torch_time, cuda_time, speedup, effective_bandwidth, tera_flops])
	results.float_format = '0.2'
	print(results)

