
from einops import repeat, einsum
from dataclasses import dataclass

import torch, math
import torch.nn.functional as F
from torch import nn

from scans.group_scan import pscan as pscan2
from scans.parallel_scan import pscan as pscan1
from einops import rearrange
from opt_einsum import contract

from flashfftconv import FlashFFTConv


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
	group = False
	ngroups = 8
	dropout = 0.1
	warp = 16
	block_size = 0

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


class LongConv(nn.Module):
	def __init__(
			self,
			H,
			L,
			channels=1,
			dropout=0.1,
			kernel_lam=0.1, 
			kernel_dropout=0,
	):
		super().__init__()
		self.H = H
		self.L = L * 2
		self.channels = channels
		self.kernel_learning_rate = kernel_learning_rate
		self.kernel_lam = kernel_lam

		self.D = nn.Parameter(torch.randn(channels, self.H))
		self.activation = F.silu()

		self.output_linear = nn.Sequential(
			nn.Linear(self.channels * self.H, 2 * self.H, bias=True),
			nn.GLU(dim=-1),
		)

		self.kernel = torch.nn.Parameter(torch.randn(self.channels, self.H, self.L) * 0.002) #(c,H,L) 


	def forward(self, u):
		L = u.size(-1)

		k = self.kernel

		# squash operator
		k = F.relu(torch.abs(k) - self.kernel_lam) * torch.sign(k)
		# use FFT to compute convolution
		y = self.flashfftconv(u.contiguous(), k.squeeze(0))
		y = y.unsqueeze(1)

		# Compute skip connection
		y = y + contract('bhl,ch->bchl', u, self.D)

		# Reshape to flatten channels
		y = rearrange(y, '... c h l -> ... (c h) l')

		y = self.activation(y)

		# Transpose for the linear
		y = y.transpose(-1, -2)
		y = self.output_linear(y)
		y = y.transpose(-1, -2)

		return y

class LongConvModel(nn.Module):

	def __init__(
		self,
		d_model=256,
		dropout=0.1,
		prenorm=False,
		**conv_kwargs,
	):
		super().__init__()

		self.flashfftconv = FlashFFTConv(config.ngroups * 2, dtype=torch.bfloat16)

		self.layer = LongConv(d_model, L=config.ngroups, dropout=dropout, **conv_kwargs)
		self.layer.flashfftconv = self.flashfftconv


	def forward(self, x):
		x_type = x.dtype
		x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
		x = (self.layer(x) + x).transpose(-1, -2)
		return x


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

		if config.group:
			self.warp = config.warp
			self.ng = config.ngroups
			self.ngs = [x * self.warp for x in range(self.ng)]
			self.hot_loop = self.group_block
			self.second_mixing = LongConvModel(self.d_in * self.d_state)
		else:
			self.hot_loop = self.vanilla_block_parallel


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

		y, latent = self.hot_loop(x, delta, A, B, C, D, latent)
		y = y * F.silu(res)

		return self.out_proj(y), latent


	def vanilla_block_seq(self, u, delta, A, B, C, D, latent=None):
		b, l, d_in = u.shape
		n = A.shape[1]

		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

		x = torch.zeros((b, d_in, n), device=deltaA.device)
		ys = []
		for i in range(l):
			x = deltaA[:, i] * x + deltaB_u[:, i]
			y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
			ys.append(y)

		y = torch.stack(ys, dim=1)
		
		y = y + u * D

		return y, None


	def vanilla_block_parallel(self, u, delta, A, B, C, D, latent=None):
		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

		hs = pscan1(deltaA, deltaB_u)
		y = (hs @ C.unsqueeze(-1)).squeeze()
		y = y + u * D

		return y, None


	def group_block(self, u, delta, A, B, C, D, latent=None):
		b, l, d_in = u.shape
		n = A.shape[1]

		deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
		deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

		if self.idx > 0:
			deltaB_u = deltaB_u.view(b, self.ng, self.warp, d_in, n)

			deltaB_u = torch.cat(
				(deltaB_u[:, :1],
				torch.cat(
					((deltaB_u[:, 1:, 0] + (deltaA[:, self.ngs[1:]] * latent[:, :-1])).unsqueeze(2),
					deltaB_u[:, 1:, 1:]), dim=2)
				),
			dim=1).view(b, self.ng * self.warp, d_in, n)


		latent = pscan2(deltaA, deltaB_u)

		y = (latent @ C.unsqueeze(-1)).squeeze().view(b, l, d_in) + u * D

		# return (y,
		# 	torch.fft.fft2(
		# 		latent.view(b, self.ng, self.warp, d_in, n)[:, :, -1].float(),
		# 	dim=(-1, -2)).real
		# )
		# return (y, None)
		return (y,
			self.second_mixing(
				latent.view(b, self.ng, self.warp, d_in, n)[:, :, -1].view(b, self.ng, d_in * n).to(torch.bfloat16)
			).view(b, self.ng, d_in, n)
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
	def __init__(self):
		super().__init__()
		self.blocks = nn.ModuleList([Block(idx) for idx in range(config.nlayers)])


	def forward(self, x):
		latent = None
		for i, block in enumerate(self.blocks):
			x, latent = block(x, latent)
		return x
