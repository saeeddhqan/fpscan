from __future__ import annotations

import importlib
import os
import sys
from typing import Optional

import torch


__all__ = ["pscan", "scan_forward", "PScan"]


_EXT_CANDIDATES = ("fpscan._C", "something_cool", "something_weird")

_CSRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
_ext = None


def _load_extension():
    global _ext
    if _ext is not None:
        return _ext

    if os.path.isdir(_CSRC_DIR) and _CSRC_DIR not in sys.path:
        sys.path.append(_CSRC_DIR)

    errors = []
    for name in _EXT_CANDIDATES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # not built / wrong arch / import error
            errors.append(f"  {name}: {exc}")
            continue
        if hasattr(module, "fullscan_forward"):
            _ext = module
            return _ext
        errors.append(f"  {name}: imported but has no 'fullscan_forward'")

    raise ImportError(
        "Could not import the fpscan CUDA extension. Build it from the csrc/ "
        "directory first (e.g. `cd csrc && python setup.py build_ext "
        "--inplace`). Tried:\n" + "\n".join(errors)
    )


def scan_forward(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    ext = _load_extension()
    if out is None:
        out = torch.empty_like(tokens)
    ext.fullscan_forward(gates, tokens, out, reverse)
    return out


class PScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        out = scan_forward(gates, tokens)
        ctx.save_for_backward(gates, out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gates, out = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        shifted_gates = torch.cat(
            [gates, torch.ones_like(gates[:, :, :1])], dim=-1
        )[:, :, 1:].contiguous()
        grad_tokens = scan_forward(shifted_gates, grad_output, reverse=True)

        shifted_out = torch.cat(
            [torch.zeros_like(out[:, :, :1]), out], dim=-1
        )[:, :, :-1]
        grad_gates = shifted_out * grad_tokens
        return grad_gates, grad_tokens


def pscan(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return PScan.apply(gates, tokens)
