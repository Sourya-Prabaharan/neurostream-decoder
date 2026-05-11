"""Generate demo session data + trained-model outputs for the dashboard.

Writes JSON artifacts to ``data/`` that the Express backend serves:

    data/sessions.json        – overview of available demo sessions
    data/session_<id>.json    – per-session preview (signal traces, spikes,
                                 cursor, labels, decoded predictions,
                                 streaming windows, telemetry)
    data/benchmark.json       – throughput + latency percentiles
    data/architecture.json    – pipeline metadata for the UI diagram
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.neurostream.signal_gen import GeneratorConfig, generate_session
from engine.neurostream.processing import (
    BandpassState,
    FeatureConfig,
    bandpass_apply,
    compute_features,
    detect_spikes,
    window_labels,
)
from engine.neurostream.models import ModelConfig, build_model, train_decoder
from engine.neurostream.streaming import StreamConfig, StreamingDecoder
from engine.neurostream.device import device_report, get_device
from engine.neurostream.benchmarks import run_benchmark


DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    n = (arr.shape[0] // factor) * factor
    if n == 0:
        return arr
    return arr[:n].reshape(n // factor, factor, *arr.shape[1:]).mean(axis=1)


def build_session(session_id: str, task: str, seed: int, duration: float, model_kind: str):
    print(f"[{session_id}] generating {task} session ({duration}s) …")
    gen_cfg = GeneratorConfig(
        n_channels=24,
        sample_rate=1000,
        duration_sec=duration,
        seed=seed,
        task=task,
    )
    session = generate_session(gen_cfg)
    feat_cfg = FeatureConfig()

    # Filter once for visualization (preview only).
    bp = BandpassState.create(
        feat_cfg.bandpass_low_hz, feat_cfg.bandpass_high_hz, gen_cfg.sample_rate, gen_cfg.n_channels
    )
    filtered = bandpass_apply(bp, session.signal)
    spikes_view = detect_spikes(filtered, feat_cfg.spike_threshold_sigma)

    # Train decoder.
    feats = compute_features(session.signal, feat_cfg, gen_cfg.sample_rate)
    labels = window_labels(session.labels, feat_cfg)
    n_classes = len(session.label_names)
    model_cfg = ModelConfig(input_dim=feats.shape[1], n_classes=n_classes, kind=model_kind)
    device = get_device()
    model = build_model(model_cfg)
    t0 = time.perf_counter()
    losses = train_decoder(
        model, torch.from_numpy(feats), torch.from_numpy(labels),
        device=device, epochs=6,
    )
    train_wall = time.perf_counter() - t0

    # Streaming inference.
    stream_cfg = StreamConfig(
        fs=gen_cfg.sample_rate,
        window_samples=feat_cfg.window_samples,
        hop_samples=feat_cfg.hop_samples,
    )
    decoder = StreamingDecoder(
        model, feat_dim=feats.shape[1], n_channels=gen_cfg.n_channels,
        cfg=stream_cfg, device=device,
    )
    eval_session = generate_session(
        GeneratorConfig(
            n_channels=gen_cfg.n_channels,
            sample_rate=gen_cfg.sample_rate,
            duration_sec=duration,
            seed=seed + 1000,
            task=task,
        )
    )
    t0 = time.perf_counter()
    res = decoder.stream(eval_session.signal)
    stream_wall = time.perf_counter() - t0

    eval_labels = window_labels(eval_session.labels, feat_cfg)
    correct = int(np.sum(np.asarray(res.pred_classes) == eval_labels[: len(res.pred_classes)]))
    accuracy = correct / max(1, len(res.pred_classes))

    # Down-sample for the UI (avoid sending 30k samples).
    ds_factor = max(1, gen_cfg.sample_rate // 200)  # ~200 Hz for display
    signal_preview = downsample(filtered[:, :6], ds_factor)   # first 6 channels
    time_preview = downsample(session.time.reshape(-1, 1), ds_factor).flatten()

    # Spike raster: per-channel list of spike times in seconds (cap to 200).
    raster = []
    for ch in range(min(gen_cfg.n_channels, 16)):
        idx = np.flatnonzero(spikes_view[:, ch])[:200]
        raster.append([float(i / gen_cfg.sample_rate) for i in idx])

    return {
        "id": session_id,
        "task": task,
        "model_kind": model_kind,
        "label_names": session.label_names,
        "config": {
            "n_channels": gen_cfg.n_channels,
            "sample_rate": gen_cfg.sample_rate,
            "duration_sec": gen_cfg.duration_sec,
            "feature_window_ms": int(1000 * feat_cfg.window_samples / gen_cfg.sample_rate),
            "feature_hop_ms": int(1000 * feat_cfg.hop_samples / gen_cfg.sample_rate),
            "spike_threshold_sigma": feat_cfg.spike_threshold_sigma,
        },
        "signal_preview": {
            "time": [round(float(x), 4) for x in time_preview.tolist()],
            "channels": [[round(float(v), 4) for v in row] for row in signal_preview.T.tolist()],
        },
        "cursor": [
            {"t": round(float(t), 4), "x": round(float(x), 4), "y": round(float(y), 4)}
            for t, x, y in zip(time_preview, downsample(eval_session.cursor[:, 0:1], ds_factor).flatten(),
                               downsample(eval_session.cursor[:, 1:2], ds_factor).flatten())
        ],
        "spike_raster": raster,
        "predictions": [
            {
                "t": round(float(t), 4),
                "class": int(c),
                "label": session.label_names[int(c)],
                "confidence": round(float(p), 4),
                "latency_ms": round(float(l), 3),
                "spike_count": int(s),
            }
            for t, c, p, l, s in zip(
                res.timestamps, res.pred_classes, res.confidences,
                res.latencies_ms, res.spike_counts,
            )
        ],
        "training": {
            "epochs": len(losses),
            "loss_history": [round(float(x), 4) for x in losses],
            "wall_seconds": round(train_wall, 3),
        },
        "streaming": {
            "n_windows": len(res.timestamps),
            "wall_seconds": round(stream_wall, 3),
            "windows_per_second": round(len(res.timestamps) / max(stream_wall, 1e-6), 1),
            "p50_latency_ms": round(res.percentile(50), 3),
            "p95_latency_ms": round(res.percentile(95), 3),
            "p99_latency_ms": round(res.percentile(99), 3),
            "mean_confidence": round(float(np.mean(res.confidences)), 4) if res.confidences else 0.0,
            "accuracy_proxy": round(accuracy, 4),
        },
    }


def main():
    print("Device report:", device_report().to_dict())

    sessions_meta = []
    configs = [
        ("movement_gru", "movement", 7, 30.0, "gru"),
        ("speech_gru", "speech", 19, 30.0, "gru"),
        ("movement_transformer", "movement", 31, 20.0, "transformer"),
    ]
    for sid, task, seed, dur, kind in configs:
        sess = build_session(sid, task, seed, dur, kind)
        with open(DATA_DIR / f"session_{sid}.json", "w") as f:
            json.dump(sess, f)
        sessions_meta.append({
            "id": sid,
            "task": task,
            "model_kind": kind,
            "duration_sec": dur,
            "n_channels": sess["config"]["n_channels"],
            "label_names": sess["label_names"],
            "streaming": sess["streaming"],
        })

    with open(DATA_DIR / "sessions.json", "w") as f:
        json.dump({"sessions": sessions_meta, "device": device_report().to_dict()}, f)

    print("Running benchmark …")
    bench = run_benchmark(duration_sec=20.0, task="movement", model_kind="gru", epochs=4)
    with open(DATA_DIR / "benchmark.json", "w") as f:
        json.dump(bench, f, default=float)

    architecture = {
        "stages": [
            {"name": "Multi-channel recording",
             "detail": "Synthetic 24-channel extracellular signal @ 1 kHz",
             "input": "Microelectrode array", "output": "Raw signal (T, C)"},
            {"name": "Streaming bandpass (IIR biquad)",
             "detail": "5–200 Hz local-field band; per-channel z-state preserved across windows",
             "input": "Raw signal", "output": "Filtered signal"},
            {"name": "Threshold spike detection",
             "detail": "MAD-based 4σ threshold, 8-sample refractory",
             "input": "Filtered signal", "output": "Spike events (T, C)"},
            {"name": "Window features",
             "detail": "50 ms windows w/ 25 ms hop — spike counts, RMS, MAV per channel",
             "input": "Filtered + spikes", "output": "Feature vector (3·C)"},
            {"name": "Sequence decoder",
             "detail": "GRU (default) or 2-layer Transformer encoder on PyTorch; CUDA when available",
             "input": "Feature stream", "output": "Per-window class logits"},
            {"name": "Streaming inference",
             "detail": "Recurrent state preserved across windows; latency tracked per step",
             "input": "Features", "output": "Decoded movement / command + confidence"},
        ],
    }
    with open(DATA_DIR / "architecture.json", "w") as f:
        json.dump(architecture, f)

    print("Done. Wrote:", sorted(os.listdir(DATA_DIR)))


if __name__ == "__main__":
    main()
