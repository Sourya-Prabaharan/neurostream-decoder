"""Streaming pipeline benchmark.

Runs the async pipeline at several configurations:

* 1 stream / 160 Hz / real PhysioNet recording, realtime pacing
* 1 stream / 160 Hz / synthetic recording, max-throughput pacing
* N concurrent streams (default 4) / 160 Hz / synthetic, realtime pacing
* 1 stream / 256 Hz / synthetic, realtime pacing

Writes ``data/stream_benchmark.json`` and the raw record arrays needed
for the latency histogram / throughput / inference timing charts.

Usage:
    python -m engine.scripts.stream_benchmark
    python -m engine.scripts.stream_benchmark --concurrent 4 --duration 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from engine.neurostream.realdata import DATASET_FS, load_run
from engine.neurostream.streaming_api.classifier import CNNConfig, EEGNetSmall
from engine.neurostream.streaming_api.pipeline import (
    PipelineConfig,
    run_concurrent_streams,
    run_single_stream,
)


def synth_eeg(seconds: float, fs: int, n_channels: int, seed: int = 0) -> np.ndarray:
    """Pink-ish multi-channel noise with a 10 Hz alpha bump on a few channels.

    Used when we need a longer / higher-rate trace than the downloaded
    real run can supply. Not a substitute for real EEG — used only for
    throughput / scaling measurements that don't depend on label quality.
    """
    rng = np.random.default_rng(seed)
    T = int(seconds * fs)
    t = np.arange(T) / fs
    sig = rng.standard_normal((T, n_channels)).astype(np.float32) * 20.0
    bump = (10.0 * np.sin(2 * np.pi * 10.0 * t)).astype(np.float32)
    sig[:, :8] += bump[:, None]
    return sig


def _condense(summary: dict) -> dict:
    """Drop large arrays before serialising to keep the json small."""
    s = {k: v for k, v in summary.items() if k != "per_stream_results"}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrent", type=int, default=4)
    p.add_argument("--duration", type=float, default=10.0,
                   help="synthetic stream seconds per scenario")
    p.add_argument("--out", default=str(ROOT / "data" / "stream_benchmark.json"))
    p.add_argument("--records-out", default=str(ROOT / "data" / "stream_benchmark_records.npz"))
    args = p.parse_args()

    n_ch = 64
    window_samples = 160
    hop = 40
    cfg160 = PipelineConfig(
        fs=160, n_channels=n_ch,
        window_samples=window_samples, hop_samples=hop, packet_samples=16,
    )
    cfg256 = PipelineConfig(
        fs=256, n_channels=n_ch,
        window_samples=window_samples, hop_samples=hop, packet_samples=32,
    )

    print(f"build model n_channels={n_ch} window={window_samples}")
    model = EEGNetSmall(CNNConfig(n_channels=n_ch, window_samples=window_samples, n_classes=3))
    model.eval()

    scenarios = {}
    record_pool = {}

    # 1. Real PhysioNet EEG, realtime, 1 stream.
    edf = ROOT / "data" / "raw_eeg" / "S001R03.edf"
    if edf.exists():
        sig, _, info = load_run(edf)
        secs = min(args.duration, info["duration_s"])
        n = int(secs * DATASET_FS)
        print(f"[1] real EEG, 1 stream, realtime, {secs:.1f}s ({n} samples)")
        res = asyncio.run(run_single_stream(
            sig[:n].astype(np.float32), cfg160, model,
            stream_id="real", realtime=True, backend="in-memory",
        ))
        scenarios["real_1stream_160hz_realtime"] = _condense(res)

    # 2. Synthetic, 1 stream, max throughput (realtime=False).
    print(f"[2] synthetic, 1 stream, max-throughput, {args.duration:.1f}s")
    sig = synth_eeg(args.duration, 160, n_ch, seed=1)
    res = asyncio.run(run_single_stream(
        sig, cfg160, model, stream_id="t0",
        realtime=False, backend="in-memory",
    ))
    scenarios["synth_1stream_160hz_maxthroughput"] = _condense(res)
    # Save raw records for chart generation.
    # We extract from the metrics inside the summary if present.

    # 3. N concurrent streams, realtime, synthetic.
    print(f"[3] synthetic, {args.concurrent} concurrent streams, realtime, "
          f"{args.duration:.1f}s")
    sigs = [synth_eeg(args.duration, 160, n_ch, seed=i + 10)
            for i in range(args.concurrent)]
    res = asyncio.run(run_concurrent_streams(
        sigs, cfg160, model, realtime=True, backend="in-memory",
    ))
    scenarios[f"synth_{args.concurrent}streams_160hz_realtime"] = _condense(res)

    # 4. Synthetic, 1 stream, 256 Hz, realtime.
    print(f"[4] synthetic, 1 stream, 256 Hz, realtime, {args.duration:.1f}s")
    sig = synth_eeg(args.duration, 256, n_ch, seed=2)
    res = asyncio.run(run_single_stream(
        sig, cfg256, model, stream_id="hf",
        realtime=True, backend="in-memory",
    ))
    scenarios["synth_1stream_256hz_realtime"] = _condense(res)

    out = {
        "config": {
            "concurrent_streams": args.concurrent,
            "duration_s": args.duration,
            "n_channels": n_ch,
            "window_samples": window_samples,
            "hop_samples": hop,
            "model": "EEGNetSmall (1D CNN, 2 conv blocks)",
            "model_param_count": sum(p.numel() for p in model.parameters()),
            "torch_version": torch.__version__,
        },
        "scenarios": scenarios,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print()
    print(f"wrote {args.out}")
    for name, s in scenarios.items():
        e2e = s.get("end_to_end_ms", {})
        inf = s.get("inference_ms", {})
        print(f"  {name}: "
              f"e2e p50/p95/p99 = {e2e.get('p50', 0):.2f}/"
              f"{e2e.get('p95', 0):.2f}/{e2e.get('p99', 0):.2f} ms  "
              f"inf p50/p95 = {inf.get('p50', 0):.2f}/{inf.get('p95', 0):.2f} ms  "
              f"wps={s.get('windows_per_second', 0):.0f}  "
              f"rss_peak={s.get('rss_mb_peak', 0):.0f} MB")


if __name__ == "__main__":
    main()
