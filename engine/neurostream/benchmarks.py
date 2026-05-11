"""Benchmark helpers — throughput + latency percentiles."""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch

from .streaming import StreamingDecoder, StreamConfig
from .signal_gen import GeneratorConfig, generate_session
from .processing import FeatureConfig, compute_features, window_labels
from .models import ModelConfig, build_model, train_decoder
from .device import get_device, device_report


def run_benchmark(
    duration_sec: float = 30.0,
    task: str = "movement",
    model_kind: str = "gru",
    epochs: int = 5,
) -> Dict:
    """End-to-end: generate → train → stream → measure."""
    device = get_device()
    rep = device_report().to_dict()

    gen_cfg = GeneratorConfig(duration_sec=duration_sec, task=task, seed=7)
    session = generate_session(gen_cfg)
    feat_cfg = FeatureConfig()

    # Training features.
    train_feats = compute_features(session.signal, feat_cfg, session.config.sample_rate)
    train_labels = window_labels(session.labels, feat_cfg)
    n_classes = len(session.label_names)

    model_cfg = ModelConfig(
        input_dim=train_feats.shape[1],
        n_classes=n_classes,
        kind=model_kind,
    )
    model = build_model(model_cfg)
    t0 = time.perf_counter()
    losses = train_decoder(
        model,
        torch.from_numpy(train_feats),
        torch.from_numpy(train_labels),
        device=device,
        epochs=epochs,
    )
    train_time = time.perf_counter() - t0

    # Streaming pass on a fresh session.
    eval_session = generate_session(
        GeneratorConfig(duration_sec=duration_sec, task=task, seed=11)
    )
    stream_cfg = StreamConfig(
        fs=eval_session.config.sample_rate,
        window_samples=feat_cfg.window_samples,
        hop_samples=feat_cfg.hop_samples,
    )
    decoder = StreamingDecoder(
        model,
        feat_dim=train_feats.shape[1],
        n_channels=eval_session.signal.shape[1],
        cfg=stream_cfg,
        device=device,
    )
    t0 = time.perf_counter()
    result = decoder.stream(eval_session.signal)
    wall = time.perf_counter() - t0

    windows = len(result.timestamps)
    throughput = windows / wall if wall > 0 else 0.0

    # Accuracy proxy: did predicted class match the *windowed* eval label?
    eval_labels = window_labels(eval_session.labels, feat_cfg)
    correct = int(np.sum(np.asarray(result.pred_classes) == eval_labels[:windows]))
    acc = correct / max(1, windows)

    return {
        "device_report": rep,
        "training": {
            "epochs": epochs,
            "loss_history": losses,
            "wall_seconds": train_time,
            "model_kind": model_kind,
        },
        "streaming": {
            "n_windows": windows,
            "wall_seconds": wall,
            "windows_per_second": throughput,
            "p50_latency_ms": result.percentile(50),
            "p95_latency_ms": result.percentile(95),
            "p99_latency_ms": result.percentile(99),
            "accuracy_proxy": acc,
        },
        "task": task,
        "label_names": session.label_names,
    }
