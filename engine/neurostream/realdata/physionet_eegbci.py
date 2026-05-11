"""PhysioNet EEG Motor Movement/Imagery Database (eegmmidb) loader.

Dataset facts (from physionet.org/content/eegmmidb/1.0.0):
    * 109 subjects
    * 64 channels, 160 Hz sample rate
    * BCI2000 standard EDF+ files
    * Annotations:
        - T0: rest
        - T1: motion of left fist (runs 3, 7, 11) or both fists (4, 8, 12)
        - T2: motion of right fist (runs 3, 7, 11) or both feet (4, 8, 12)

Runs 3, 7, 11 use 4-class motor imagery (rest, left fist, right fist,
optionally imagined variants). We load runs 3 and 4 by default which give
us 3-4 label classes per file before merging.

Network access: a single ~2.5 MB EDF per run. Cached to ``data/raw_eeg``.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

import numpy as np

DATASET_FS = 160
DATASET_N_CHANNELS = 64
PHYSIONET_BASE = "https://physionet.org/files/eegmmidb/1.0.0"

# Labels used after parsing T0/T1/T2 events. We intentionally collapse the
# left/right fist into 3 classes (rest / left / right) so the small CNN
# baseline has a meaningful signal to learn.
LABELS = ["rest", "left_fist", "right_fist"]


def _run_url(subject: int, run: int) -> str:
    sid = f"S{subject:03d}"
    fname = f"{sid}R{run:02d}.edf"
    return f"{PHYSIONET_BASE}/{sid}/{fname}"


def download_run(subject: int, run: int, cache_dir: Path) -> Path:
    """Download one EDF run to the cache dir. Returns the local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sid = f"S{subject:03d}"
    fname = f"{sid}R{run:02d}.edf"
    dst = cache_dir / fname
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = _run_url(subject, run)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    dst.write_bytes(data)
    return dst


def load_run(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read an EDF file and return ``(signal, labels, info)``.

    * ``signal`` – (T, C) float32, in microvolts (pyedflib native units).
    * ``labels`` – (T,) int64, sample-wise class index into ``LABELS``.
    * ``info``   – metadata (fs, channels, annotations).
    """
    import pyedflib

    f = pyedflib.EdfReader(str(path))
    try:
        n_channels = f.signals_in_file
        fs = int(f.getSampleFrequency(0))
        n_samples = f.getNSamples()[0]
        sig = np.zeros((n_samples, n_channels), dtype=np.float32)
        for c in range(n_channels):
            sig[:, c] = f.readSignal(c).astype(np.float32)
        ch_names = list(f.getSignalLabels())
        ann_t, ann_d, ann_lbl = f.readAnnotations()
    finally:
        f._close()  # type: ignore[attr-defined]

    labels = np.zeros(n_samples, dtype=np.int64)
    annotations = []
    for t, d, lbl in zip(ann_t, ann_d, ann_lbl):
        # lbl can be bytes, str, or np.str_ depending on pyedflib version.
        if isinstance(lbl, bytes):
            lbl = lbl.decode()
        lbl = str(lbl)
        s = int(round(t * fs))
        e = min(n_samples, int(round((t + d) * fs)))
        cls = 0
        if lbl == "T1":
            cls = 1
        elif lbl == "T2":
            cls = 2
        labels[s:e] = cls
        annotations.append({"t": float(t), "d": float(d), "label": lbl, "class": cls})

    info = {
        "path": str(path),
        "fs": fs,
        "n_channels": n_channels,
        "channel_names": ch_names,
        "duration_s": n_samples / fs,
        "n_samples": n_samples,
        "annotations": annotations,
        "labels_used": LABELS,
    }
    return sig, labels, info


def run_to_windows(
    signal: np.ndarray,
    labels: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cut sliding windows from a run for offline CNN training.

    Returns ``(X, y)`` with X shape (N, C, T) and y shape (N,). The
    per-window label is the majority sample label inside the window.
    """
    T, C = signal.shape
    n_win = max(0, 1 + (T - window_samples) // hop_samples)
    X = np.zeros((n_win, C, window_samples), dtype=np.float32)
    y = np.zeros(n_win, dtype=np.int64)
    for i in range(n_win):
        a = i * hop_samples
        b = a + window_samples
        seg = signal[a:b]
        # Per-channel z-score so all windows live on similar scale before
        # they reach the CNN.
        mu = seg.mean(axis=0, keepdims=True)
        sd = seg.std(axis=0, keepdims=True) + 1e-6
        X[i] = ((seg - mu) / sd).T
        vals, counts = np.unique(labels[a:b], return_counts=True)
        y[i] = int(vals[np.argmax(counts)])
    return X, y
