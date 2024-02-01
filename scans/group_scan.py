from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

cuda_source = (Path(__file__).parent / 'csrc' / 'myscan.cu').read_text()
cpp_source = (Path(__file__).parent / 'csrc' / 'main.cpp').read_text()

myscan = load_inline(
    name='myscan',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['myscan_forward'],
    verbose=False,
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--ptxas-options=-v",
        "-lineinfo",
        "--fmad", "false",
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
)
myscan_forward = myscan.myscan_forward


class PScan(torch.autograd.Function):
	@staticmethod
	def pscan(A, X):
		myscan_forward(
			A.mT.contiguous(),
			X.mT.contiguous(),
		)
		# return out.mT.transpose(2, 1)

	@staticmethod
	def forward(ctx, A_in, X_in):

		A = A_in.clone()
		X = X_in.clone()

		A = A.transpose(2, 1)
		X = X.transpose(2, 1)

		myscan_forward(
			A.mT.contiguous(),
			X.mT.contiguous(),
		)
		# ctx.save_for_backward(A_in, X)

		return X.transpose(2, 1)
	
	@staticmethod
	def backward(ctx, grad_output_in):

		A_in, X = ctx.saved_tensors

		A = A_in.clone()

		A = A.transpose(2, 1)
		A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
		grad_output_b = grad_output_in.transpose(2, 1)

		grad_output_b = grad_output_b.flip(2)
		PScan.pscan(A, grad_output_b)
		grad_output_b = grad_output_b.flip(2)

		Q = torch.zeros_like(X)
		Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

		return Q.transpose(2, 1), grad_output_b.transpose(2, 1)
	
pscan = PScan.apply
