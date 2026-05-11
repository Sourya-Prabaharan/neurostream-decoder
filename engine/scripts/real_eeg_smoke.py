"""Smoke test: load a real PhysioNet EEG run, train the small CNN on it,
   stream it back through the async pipeline, and write a JSON report.

Outputs ``data/real_eeg_smoke.json``. Exit code is non-zero on failure.
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

from engine.neurostream.realdata import (
    DATASET_FS,
    download_run,
    load_run,
    run_to_windows,
)
from engine.neurostream.streaming_api.classifier import (
    CNNConfig,
    EEGNetSmall,
    evaluate,
    train_classifier,
)
from engine.neurostream.streaming_api.pipeline import (
    PipelineConfig,
    run_single_stream,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--run", type=int, default=3)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--out", default=str(ROOT / "data" / "real_eeg_smoke.json"))
    args = p.parse_args()

    cache = ROOT / "data" / "raw_eeg"
    print(f"[1/4] download S{args.subject:03d}R{args.run:02d}.edf -> {cache}")
    t0 = time.perf_counter()
    edf = download_run(args.subject, args.run, cache)
    t_dl = time.perf_counter() - t0

    print(f"[2/4] load + window")
    sig, labels, info = load_run(edf)
    window_samples = DATASET_FS  # 1 second
    hop = DATASET_FS // 4         # 250 ms
    X, y = run_to_windows(sig, labels, window_samples, hop)
    # Random 80/20 split — temporal split here would concentrate one class
    # on each side because annotations are large contiguous blocks.
    rng = np.random.default_rng(0)
    idx = rng.permutation(X.shape[0])
    n_tr = int(0.8 * X.shape[0])
    X_tr, y_tr = X[idx[:n_tr]], y[idx[:n_tr]]
    X_te, y_te = X[idx[n_tr:]], y[idx[n_tr:]]
    n_classes = int(max(y.max() + 1, 2))

    print(f"[3/4] train CNN ({args.epochs} epochs) on {X_tr.shape[0]} windows, "
          f"{n_classes} classes")
    model = EEGNetSmall(CNNConfig(
        n_channels=info["n_channels"], window_samples=window_samples,
        n_classes=n_classes,
    ))
    train_res = train_classifier(model, X_tr, y_tr, epochs=args.epochs, batch_size=32)
    test_metrics = evaluate(model, X_te, y_te)
    # Per-class accuracy / chance baseline.
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te).float())
        pred = logits.argmax(-1).numpy()
    per_class = {}
    for c in range(n_classes):
        mask = y_te == c
        if mask.any():
            per_class[int(c)] = float((pred[mask] == c).mean())
    chance = max(np.bincount(y_te) / max(len(y_te), 1))

    print(f"[4/4] stream the full run through the async pipeline (realtime=True)")
    cfg = PipelineConfig(
        fs=DATASET_FS,
        n_channels=info["n_channels"],
        window_samples=window_samples,
        hop_samples=hop,
        packet_samples=16,
    )
    # Cap the streamed portion at 30s of EEG so the smoke is fast (~30 s
    # wall time in realtime mode).
    max_samples = 30 * DATASET_FS
    stream_sig = sig[:max_samples].astype(np.float32)
    pipe = asyncio.run(run_single_stream(
        stream_sig, cfg, model, stream_id="real",
        realtime=True, backend="in-memory",
    ))

    out = {
        "source": "physionet eegmmidb v1.0.0",
        "subject": args.subject,
        "run": args.run,
        "dataset": {
            "fs": info["fs"],
            "n_channels": info["n_channels"],
            "duration_s": info["duration_s"],
            "n_samples": info["n_samples"],
            "n_annotations": len(info["annotations"]),
            "labels": info["labels_used"],
        },
        "training": {
            "epochs": args.epochs,
            "n_train_windows": int(X_tr.shape[0]),
            "n_test_windows": int(X_te.shape[0]),
            "history": train_res["history"],
            "final_train_loss": train_res["history"][-1]["loss"],
            "final_train_acc": train_res["history"][-1]["train_acc"],
        },
        "evaluation": {
            "test_accuracy": test_metrics["accuracy"],
            "chance_baseline": float(chance),
            "per_class_accuracy": per_class,
        },
        "streaming": pipe,
        "timings": {
            "download_seconds": t_dl,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print()
    print(f"wrote {args.out}")
    print(f"  test_acc={test_metrics['accuracy']:.3f}  chance={chance:.3f}")
    s = pipe
    print(f"  streamed {s['n_windows']} windows, end-to-end p50/p95/p99 ms = "
          f"{s['end_to_end_ms']['p50']:.2f} / {s['end_to_end_ms']['p95']:.2f} / "
          f"{s['end_to_end_ms']['p99']:.2f}")
    print(f"  inference p50/p95/p99 ms = "
          f"{s['inference_ms']['p50']:.2f} / {s['inference_ms']['p95']:.2f} / "
          f"{s['inference_ms']['p99']:.2f}")
    print(f"  windows/s = {s['windows_per_second']:.1f}, "
          f"packets/s = {s['packets_per_second']:.1f}, "
          f"RSS peak = {s['rss_mb_peak']:.1f} MB")


if __name__ == "__main__":
    main()
