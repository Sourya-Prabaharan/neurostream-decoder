"""Generate matplotlib charts from the streaming benchmark JSON.

Outputs (all into ``docs/assets/``):

* ``latency_histogram.png``    – end-to-end latency distribution per scenario
* ``throughput.png``           – windows/sec over wall-clock for each scenario
* ``inference_timing.png``     – stage-by-stage timing breakdown (bar chart)
* ``pipeline_architecture.png`` – simple block diagram of the pipeline

Run after ``stream_benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def _scenario_label(name: str) -> str:
    return (name.replace("_", " ")
                .replace("160hz", "160 Hz")
                .replace("256hz", "256 Hz"))


def latency_histogram(bench: dict, out: Path) -> None:
    """Render the *summary* latency percentiles as a horizontal bar chart.

    We don't dump the per-record array (would bloat the json), so we
    plot p50/p95/p99 stage by stage. This is what the README cites
    directly so it has to be readable, not pretty.
    """
    scenarios = bench["scenarios"]
    names = list(scenarios.keys())
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 2))
    y = np.arange(len(names))
    width = 0.27
    p50 = [scenarios[n]["end_to_end_ms"]["p50"] for n in names]
    p95 = [scenarios[n]["end_to_end_ms"]["p95"] for n in names]
    p99 = [scenarios[n]["end_to_end_ms"]["p99"] for n in names]
    ax.barh(y - width, p50, width, label="p50", color="#4c78a8")
    ax.barh(y, p95, width, label="p95", color="#f58518")
    ax.barh(y + width, p99, width, label="p99", color="#e45756")
    ax.set_yticks(y)
    ax.set_yticklabels([_scenario_label(n) for n in names], fontsize=9)
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_title("End-to-end latency by scenario (lower is better)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def throughput_chart(bench: dict, out: Path) -> None:
    scenarios = bench["scenarios"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, s in scenarios.items():
        snaps = s.get("snapshots", [])
        if not snaps:
            continue
        ts = [sn["t"] for sn in snaps]
        wps = [sn["windows_per_sec"] for sn in snaps]
        ax.plot(ts, wps, label=_scenario_label(name), linewidth=1.6)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("windows / sec")
    ax.set_title("Streaming throughput over time (per scenario)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def inference_timing(bench: dict, out: Path) -> None:
    scenarios = bench["scenarios"]
    names = list(scenarios.keys())
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(names) + 2))
    y = np.arange(len(names))
    # Stack queue / preprocess / inference (p50 each).
    qw = np.array([scenarios[n]["queue_wait_ms"]["p50"] for n in names])
    pp = np.array([scenarios[n]["preprocess_ms"]["p50"] for n in names])
    inf = np.array([scenarios[n]["inference_ms"]["p50"] for n in names])
    ax.barh(y, qw, color="#9ecae1", label="queue wait (p50)")
    ax.barh(y, pp, left=qw, color="#fdae6b", label="preprocess (p50)")
    ax.barh(y, inf, left=qw + pp, color="#74c476", label="inference (p50)")
    ax.set_yticks(y)
    ax.set_yticklabels([_scenario_label(n) for n in names], fontsize=9)
    ax.set_xlabel("milliseconds")
    ax.set_title("Per-window stage timing breakdown (p50)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def pipeline_architecture(out: Path) -> None:
    """A simple block diagram drawn with patches."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    boxes = [
        (0.2, "EEG source\n(PhysioNet EDF\nor synthetic)"),
        (1.9, "Producer\n(async, paces\nto fs)"),
        (3.6, "Stream\n(Redis Streams /\nin-memory queue)"),
        (5.5, "Ring buffer\n(per stream,\nbounded)"),
        (7.2, "Preprocess\nbandpass + z-score\n+ rFFT band power"),
        (8.9, "CNN inference\n(EEGNetSmall,\n~6k params)"),
        (10.3, "Metrics\n(latency / RSS /\nthroughput)"),
    ]
    last_x = None
    for x, label in boxes:
        rect = patches.FancyBboxPatch((x, 1.8), 1.3, 1.4,
                                      boxstyle="round,pad=0.08",
                                      linewidth=1.2,
                                      edgecolor="#444",
                                      facecolor="#f0f4ff")
        ax.add_patch(rect)
        ax.text(x + 0.65, 2.5, label, ha="center", va="center", fontsize=8)
        if last_x is not None:
            ax.annotate("", xy=(x, 2.5), xytext=(last_x + 1.3, 2.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2, color="#444"))
        last_x = x
    ax.text(5.5, 0.6,
            "single-process FastAPI app  |  one consumer task per stream  |  shared PyTorch model",
            ha="center", fontsize=9, color="#444")
    ax.text(5.5, 4.2, "NeuroStream streaming pipeline",
            ha="center", fontsize=12, fontweight="bold")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default=str(ROOT / "data" / "stream_benchmark.json"))
    p.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    args = p.parse_args()
    bench = json.loads(Path(args.bench).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    latency_histogram(bench, out / "latency_histogram.png")
    throughput_chart(bench, out / "throughput.png")
    inference_timing(bench, out / "inference_timing.png")
    pipeline_architecture(out / "pipeline_architecture.png")
    print(f"wrote: {sorted(p.name for p in out.glob('*.png'))}")


if __name__ == "__main__":
    main()
