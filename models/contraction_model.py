
from einops import repeat, einsum
from dataclasses import dataclass

import torch, math
import torch.nn.functional as F
from torch import nn

from models.scans.pscan1 import pscan as pscan1
from models.scans.pscan_cuda import pscan as pscan3
from einops import rearrange
from opt_einsum import contract


@dataclass
class ConfigMamba:
	dim = 64
	nlayers = 2
	d_state = 16
	expand = 2
	dt_rank = 'auto'
	d_conv = 4 
	d_inner = 0
	conv_bias = True
	bias = False
	group = False
	ngroups = 8
	dropout = 0.1
	cuda_version = False

	def __post_init__(self):
		self.d_inner = int(self.expand * self.dim)
		if self.dt_rank == 'auto':
			self.dt_rank = math.ceil(self.dim / 16)


config = ConfigMamba()


class RMSNorm(nn.Module):
	def __init__(self, dim, eps=1e-5):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))


	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return (output * self.weight)


class MambaBlock(nn.Module):
	def __init__(self, idx):
		super().__init__()
		self.idx = idx
		self.d_state = config.d_state
		self.d_in = config.d_inner
		self.dim = config.dim
		self.in_proj = nn.Linear(self.dim, self.d_in * 2, bias=config.bias)

		self.conv1d = nn.Conv1d(
			in_channels=self.d_in,
			out_channels=self.d_in,
			bias=config.conv_bias,
			kernel_size=config.d_conv,
			groups=self.d_in,
			padding=config.d_conv - 1,
		)

		self.x_proj = nn.Linear(self.d_in, config.dt_rank + config.d_state * 2, bias=False)
		self.dt_proj = nn.Linear(config.dt_rank, self.d_in, bias=True)
		self.out_proj = nn.Linear(self.d_in, self.dim, bias=config.bias)

		A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_in)
		self.A_log = nn.Parameter(torch.log(A))
		self.D = nn.Parameter(torch.ones(self.d_in))

		self.ng = config.ngroups

		if config.cuda_version:
			self.pscan = pscan3
		else:
			self.pscan = pscan1

		self.latent = torch.zeros(1, self.ng, self.d_in, self.d_state).to('cuda')

	def forward(self, x, latent):
		b, l, d = x.shape

		x_res = self.in_proj(x)

		x, res = x_res.split(split_size=[self.d_in, self.d_in], dim=-1)
		x = self.conv1d(x.mT)[:, :, :l]
		x = F.silu(x.mT)

		d_in, n = self.A_log.shape
		A = -torch.exp(self.A_log.float())
		D = self.D.float()
		x_dbl = self.x_proj(x)

		delta, B, C = x_dbl.split(split_size=[config.dt_rank, n, n], dim=-1)
		delta = F.softplus(self.dt_proj(delta))

		y, latent = self.block(x, delta, A, B, C, D, latent)
		y = y * F.silu(res)

		return self.out_proj(y), latent

	# def block(self, u, delta, A, B, C, D, latent=None):
	# 	n = A.shape[1]

	# 	deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
	# 	deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
	# 	b, l, d, d_in = deltaA.shape

	# 	# if self.idx == 0:
	# 	latent = self.latent.expand(b, -1, -1, -1)

	# 	y = self.pscan(deltaA, deltaB_u, latent, self.ng)
	# 	# latent = y[:, :, -1]

	# 	# y = (y.view(b, l, d, d_in) @ C.unsqueeze(-1)).squeeze() + u * D
	# 	# return (y,
	# 	# 	torch.fft.fft2(
	# 	# 		latent.float(),
	# 	# 	dim=(-1, -2)).real
	# 	# )
	# 	return u, latent

	def block(self, u, delta, A, B, C, D, latent=None):
		# not causal
		n = A.shape[1]

		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
		b, l, d, d_in = deltaA.shape

		if self.idx == 0:
			latent = self.latent.expand(b, -1, -1, -1)

		y = self.pscan(deltaA, deltaB_u, latent, self.ng)
		latent = y[:, :, -1]

		y = (y.view(b, l, d, d_in) @ C.unsqueeze(-1)).squeeze() + u * D
		return (y,
			torch.fft.fft2(
				latent.float(),
			dim=(-1, -2)).real
		)


class Block(nn.Module):
	def __init__(self, idx):
		super().__init__()
		self.ln1 = RMSNorm(config.dim)
		self.communicate = MambaBlock(idx)


	def forward(self, x, latent):
		u, latent = self.communicate(self.ln1(x), latent)
		x = u + x
		return x, latent


class Model(nn.Module):
	def __init__(self, dim, ng, cuda_version=False):
		super().__init__()
		config.dim = dim
		config.ngroups = ng
		config.cuda_version = cuda_version
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.nlayers)])


	def forward(self, x):
		latent = None
		for i, block in enumerate(self.blocks):
			x, latent = block(x, latent)
		return x
