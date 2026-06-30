

import pytest
import torch

import fpscan
from eval.baselines import parallel_scan_torch, sequential_scan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fpscan requires a CUDA device"
)

DEVICE = "cuda"

ALL_SEQLENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
HALF_SEQLENS = [32, 64, 128, 256, 512, 1024, 2048, 4096]

TOL = {
    torch.float32: dict(atol=2e-3, rtol=2e-3),
    torch.float16: dict(atol=3e-2, rtol=3e-2),
    torch.bfloat16: dict(atol=6e-2, rtol=6e-2),
}


def _make_inputs(batch, dim, seqlen, dtype, seed=0):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    gates = 0.99 + 0.01 * torch.rand(batch, dim, seqlen, generator=gen, device=DEVICE)
    tokens = torch.randn(batch, dim, seqlen, generator=gen, device=DEVICE) / seqlen
    return gates.to(dtype).contiguous(), tokens.to(dtype).contiguous()


def _dtype_seqlens():
    for seqlen in ALL_SEQLENS:
        yield torch.float32, seqlen
    for seqlen in HALF_SEQLENS:
        yield torch.float16, seqlen
        yield torch.bfloat16, seqlen


@pytest.mark.parametrize("dtype,seqlen", list(_dtype_seqlens()))
def test_forward_matches_oracle(dtype, seqlen):
    gates, tokens = _make_inputs(2, 4, seqlen, dtype, seed=seqlen)
    out = fpscan.scan_forward(gates, tokens)

    assert out.shape == tokens.shape
    assert out.dtype == dtype

    ref = sequential_scan(gates.double(), tokens.double())
    torch.testing.assert_close(out.double(), ref, **TOL[dtype])


@pytest.mark.parametrize("seqlen", ALL_SEQLENS)
def test_reverse_matches_oracle(seqlen):
    gates, tokens = _make_inputs(2, 4, seqlen, torch.float32, seed=seqlen + 1)
    out = fpscan.scan_forward(gates, tokens, reverse=True)

    ref = sequential_scan(gates.double(), tokens.double(), reverse=True)
    torch.testing.assert_close(out.double(), ref, **TOL[torch.float32])


@pytest.mark.parametrize("seqlen", [32, 256, 1024, 4096])
def test_agrees_with_torch_parallel_baseline(seqlen):
    gates, tokens = _make_inputs(2, 4, seqlen, torch.float32, seed=seqlen + 2)
    out = fpscan.scan_forward(gates, tokens)
    baseline = parallel_scan_torch(gates.double(), tokens.double())
    torch.testing.assert_close(out.double(), baseline, **TOL[torch.float32])


@pytest.mark.parametrize("seqlen", [32, 128, 512, 2048, 8192])
def test_backward_matches_oracle(seqlen):
    gates, tokens = _make_inputs(2, 4, seqlen, torch.float32, seed=seqlen + 3)

    g = gates.clone().requires_grad_()
    x = tokens.clone().requires_grad_()
    grad_seed = torch.Generator(device=DEVICE).manual_seed(seqlen + 4)
    weight = torch.randn(*tokens.shape, generator=grad_seed, device=DEVICE)

    out = fpscan.pscan(g, x)
    (out * weight).sum().backward()

    g_ref = gates.double().clone().requires_grad_()
    x_ref = tokens.double().clone().requires_grad_()
    out_ref = sequential_scan(g_ref, x_ref)
    (out_ref * weight.double()).sum().backward()

    torch.testing.assert_close(g.grad.double(), g_ref.grad, **TOL[torch.float32])
    torch.testing.assert_close(x.grad.double(), x_ref.grad, **TOL[torch.float32])


def test_non_power_of_two_seqlen_raises():
    gates, tokens = _make_inputs(1, 1, 32, torch.float32)
    bad_gates = gates[..., :30].contiguous()
    bad_tokens = tokens[..., :30].contiguous()
    with pytest.raises(RuntimeError):
        fpscan.scan_forward(bad_gates, bad_tokens)


def test_non_contiguous_raises():
    gates, tokens = _make_inputs(2, 4, 64, torch.float32)
    with pytest.raises(RuntimeError):
        fpscan.scan_forward(gates.transpose(1, 2), tokens.transpose(1, 2))
