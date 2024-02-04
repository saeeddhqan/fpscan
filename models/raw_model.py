
from einops import repeat, einsum
from dataclasses import dataclass

import torch, math
import torch.nn.functional as F
from torch import nn

from models.scans.pscan2 import pscan as pscan2
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
	pad_vocab_size_multiple = 8
	conv_bias = True
	bias = False
	dropout = 0.1

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
		self.latent = torch.zeros(1, self.d_in, self.d_state).to('cuda')

	def forward(self, x):
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

		y = self.block(x, delta, A, B, C, D)
		y = y * F.silu(res)

		return self.out_proj(y)


	def block(self, u, delta, A, B, C, D):
		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
		b, l, D, d_in = deltaA.shape
		hs = pscan2(deltaA, deltaB_u, self.latent.expand(b, -1, -1))
		y = (hs @ C.unsqueeze(-1)).squeeze()
		y = y + u * D

		return y


class Block(nn.Module):
	def __init__(self, idx):
		super().__init__()
		self.ln1 = RMSNorm(config.dim)
		self.communicate = MambaBlock(idx)


	def forward(self, x):
		u = self.communicate(self.ln1(x))
		x = u + x
		return x


class Model(nn.Module):
	def __init__(self, dim):
		super().__init__()
		config.dim = dim
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.nlayers)])


	def forward(self, x):
		for i, block in enumerate(self.blocks):
			x = block(x)
		return x
