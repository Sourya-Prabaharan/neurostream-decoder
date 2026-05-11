"""Standalone benchmark — reports throughput + latency percentiles.

Usage:
    python -m engine.scripts.benchmark --task movement --kind gru --duration 30 --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.neurostream.benchmarks import run_benchmark


def main():
    p = argparse.ArgumentParser(description="Run an end-to-end NeuroStream benchmark.")
    p.add_argument("--task", choices=["movement", "speech"], default="movement")
    p.add_argument("--kind", choices=["gru", "transformer"], default="gru")
    p.add_argument("--duration", type=float, default=20.0, help="session seconds")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--json", action="store_true", help="emit JSON only")
    args = p.parse_args()

    res = run_benchmark(
        duration_sec=args.duration,
        task=args.task,
        model_kind=args.kind,
        epochs=args.epochs,
    )

    if args.json:
        print(json.dumps(res, default=float))
        return

    s = res["streaming"]
    d = res["device_report"]
    print()
    print(f"Device       : {d['device']}  cuda={d['cuda_available']}  torch={d['torch_version']}")
    print(f"Task / Model : {args.task} / {args.kind}")
    print(f"Train wall   : {res['training']['wall_seconds']:.2f}s "
          f"(loss {res['training']['loss_history'][0]:.3f} → {res['training']['loss_history'][-1]:.3f})")
    print()
    print(f"Streaming windows     : {s['n_windows']}")
    print(f"Wall time             : {s['wall_seconds']:.3f}s")
    print(f"Throughput            : {s['windows_per_second']:.1f} windows/sec")
    print(f"Latency  p50 / p95 / p99 : "
          f"{s['p50_latency_ms']:.2f} / {s['p95_latency_ms']:.2f} / {s['p99_latency_ms']:.2f} ms")
    print(f"Accuracy proxy        : {s['accuracy_proxy'] * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()
