

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


__all__ = [
    "sequential_scan",
    "parallel_scan_torch",
    "torch_associative_scan",
    "ASSOCIATIVE_SCAN_AVAILABLE",
    "BASELINES",
]


def sequential_scan(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:

    if reverse:
        gates = gates.flip(-1)
        tokens = tokens.flip(-1)

    seqlen = tokens.shape[-1]
    out = torch.empty_like(tokens)
    acc = torch.zeros_like(tokens[..., 0])
    for t in range(seqlen):
        acc = gates[..., t] * acc + tokens[..., t]
        out[..., t] = acc

    return out.flip(-1) if reverse else out


def parallel_scan_torch(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:
    if reverse:
        gates = gates.flip(-1)
        tokens = tokens.flip(-1)

    a = gates
    b = tokens
    seqlen = a.shape[-1]

    delta = 1
    while delta < seqlen:
        a_prev = F.pad(a, (delta, 0), value=1.0)[..., :seqlen]
        b_prev = F.pad(b, (delta, 0), value=0.0)[..., :seqlen]
        b = a * b_prev + b
        a = a * a_prev
        delta *= 2

    return b.flip(-1) if reverse else b


def _resolve_associative_scan() -> Optional[Callable]:
    fn = getattr(torch, "associative_scan", None)
    if fn is not None:
        return fn
    try:
        from torch._higher_order_ops.associative_scan import associative_scan

        return associative_scan
    except Exception:
        return None


_ASSOCIATIVE_SCAN = _resolve_associative_scan()
ASSOCIATIVE_SCAN_AVAILABLE = _ASSOCIATIVE_SCAN is not None


def torch_associative_scan(
    gates: torch.Tensor,
    tokens: torch.Tensor,
    reverse: bool = False,
) -> torch.Tensor:

    if _ASSOCIATIVE_SCAN is None:
        raise RuntimeError(
            "torch.associative_scan is unavailable in this torch build "
            f"(torch {torch.__version__}); use parallel_scan_torch instead."
        )

    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return (a_r * a_l, a_r * b_l + b_r)

    try:
        out = _ASSOCIATIVE_SCAN(combine, (gates, tokens), dim=-1, reverse=reverse)
    except TypeError:
        if reverse:
            out = _ASSOCIATIVE_SCAN(
                combine, (gates.flip(-1), tokens.flip(-1)), dim=-1
            )
            return out[1].flip(-1)
        out = _ASSOCIATIVE_SCAN(combine, (gates, tokens), dim=-1)

    return out[1]

BASELINES = {
    "sequential (oracle)": sequential_scan,
    "torch parallel (hillis-steele)": parallel_scan_torch,
    "torch.associative_scan": torch_associative_scan,
}
