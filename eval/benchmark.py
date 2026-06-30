from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable

import fpscan
from eval.baselines import (
    ASSOCIATIVE_SCAN_AVAILABLE,
    parallel_scan_torch,
    torch_associative_scan,
)

DEFAULT_SEQLENS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def benchmark_fn(fn, warmup, iters):

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    return torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])


def percentiles(times):
    if times is None:
        return float("nan"), float("nan")
    p50 = torch.quantile(times, 0.50).item()
    p90 = torch.quantile(times, 0.90).item()
    return p50, p90


def make_inputs(batch, dim, seqlen, dtype, device):
    gates = 0.99 + 0.01 * torch.rand(batch, dim, seqlen, device=device)
    tokens = torch.randn(batch, dim, seqlen, device=device) / seqlen
    return gates.to(dtype).contiguous(), tokens.to(dtype).contiguous()


def effective_bandwidth_gbps(batch, dim, seqlen, dtype, p50_ms):
    itemsize = torch.empty((), dtype=dtype).element_size()
    bytes_moved = 3 * batch * dim * seqlen * itemsize
    return bytes_moved / 1e9 / (p50_ms / 1e3)


def build_methods(compile_assoc):
    methods = [
        ("fpscan", lambda g, x: fpscan.scan_forward(g, x)),
        ("torch parallel", lambda g, x: parallel_scan_torch(g, x)),
    ]
    if ASSOCIATIVE_SCAN_AVAILABLE:
        assoc = torch_associative_scan
        label = "torch.associative_scan"
        if compile_assoc:
            try:
                assoc = torch.compile(torch_associative_scan)
                label = "torch.associative_scan (compiled)"
            except Exception as exc:
                print(f"[warn] could not torch.compile associative_scan: {exc}")
        methods.append((label, lambda g, x: assoc(g, x)))
    else:
        print("[info] torch.associative_scan unavailable in this torch build; "
              "skipping that baseline.")
    return methods


def plot_latency(seqlens, latencies, names, path):
    plt.figure(figsize=(8, 5))
    for name in names:
        ys = latencies[name]
        xs = [s for s, y in zip(seqlens, ys) if y == y]  # drop nan
        ys = [y for y in ys if y == y]
        if ys:
            plt.plot(xs, ys, marker="o", label=name)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("sequence length")
    plt.ylabel("forward latency p50 (ms)")
    plt.title("Forward scan latency vs sequence length")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.clf()
    print(f"wrote {path}")


def plot_speedup(seqlens, latencies, baseline_names, path):
    plt.figure(figsize=(8, 5))
    base = latencies["fpscan"]
    for name in baseline_names:
        ys = latencies[name]
        xs = [s for s, b, y in zip(seqlens, base, ys) if b == b and y == y]
        sp = [y / b for b, y in zip(base, ys) if b == b and y == y]
        if sp:
            plt.plot(xs, sp, marker="o", label=f"vs {name}")
    plt.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
    plt.xscale("log", base=2)
    plt.xlabel("sequence length")
    plt.ylabel("speedup (baseline / fpscan)")
    plt.title("fpscan forward speedup over baselines")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.clf()
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--dtype", choices=list(DTYPES), default="float32")
    parser.add_argument("--seqlens", type=int, nargs="+", default=DEFAULT_SEQLENS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-assoc", action="store_true",
                        help="torch.compile the associative_scan baseline")
    parser.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires a CUDA device.")

    device = "cuda"
    dtype = DTYPES[args.dtype]
    methods = build_methods(args.compile_assoc)
    method_names = [name for name, _ in methods]
    baseline_names = [name for name in method_names if name != "fpscan"]

    print(f"device={torch.cuda.get_device_name()}  dtype={args.dtype}  "
          f"batch={args.batch}  dim={args.dim}  "
          f"warmup={args.warmup}  iters={args.iters}")

    table = PrettyTable()
    table.field_names = (
        ["seqlen"]
        + [f"{n} p50 (ms)" for n in method_names]
        + [f"{n} p90 (ms)" for n in method_names]
        + ["fpscan GB/s"]
        + [f"speedup vs {n}" for n in baseline_names]
    )
    table.float_format = "0.4"

    latencies = {name: [] for name in method_names}
    disabled = {}

    for seqlen in args.seqlens:
        gates, tokens = make_inputs(args.batch, args.dim, seqlen, dtype, device)
        p50s, p90s = {}, {}
        for name, fn in methods:
            if name in disabled:
                p50s[name] = p90s[name] = float("nan")
                latencies[name].append(float("nan"))
                continue
            try:
                t = benchmark_fn(
                    lambda g=gates, x=tokens, f=fn: f(g, x), args.warmup, args.iters
                )
            except Exception as exc:
                disabled[name] = str(exc)
                print(f"    [disabled] {name}: {exc}")
                p50s[name] = p90s[name] = float("nan")
                latencies[name].append(float("nan"))
                continue
            p50, p90 = percentiles(t)
            p50s[name], p90s[name] = p50, p90
            latencies[name].append(p50)

        fp_p50 = p50s["fpscan"]
        gbps = (effective_bandwidth_gbps(args.batch, args.dim, seqlen, dtype, fp_p50)
                if fp_p50 == fp_p50 else float("nan"))
        speedups = [
            (p50s[n] / fp_p50) if (p50s[n] == p50s[n] and fp_p50 == fp_p50) else float("nan")
            for n in baseline_names
        ]

        table.add_row(
            [seqlen]
            + [p50s[n] for n in method_names]
            + [p90s[n] for n in method_names]
            + [gbps]
            + speedups
        )
        print(f"[{seqlen}] fpscan p50={fp_p50:.4f}ms  "
              + "  ".join(f"{n}x{s:.2f}" for n, s in zip(baseline_names, speedups)))

    print(table)

    os.makedirs(args.outdir, exist_ok=True)
    plot_latency(args.seqlens, latencies, method_names,
                 os.path.join(args.outdir, "latency_vs_seqlen.png"))
    plot_speedup(args.seqlens, latencies, baseline_names,
                 os.path.join(args.outdir, "speedup_vs_seqlen.png"))


if __name__ == "__main__":
    main()
