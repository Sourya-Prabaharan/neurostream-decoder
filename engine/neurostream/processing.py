"""Signal processing — streaming-friendly filters, spike detection, features.

Filters are implemented as biquad IIR sections processed sample-by-sample so
they work identically on streaming windows and on full sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


# ---------------------------------------------------------------------------
# IIR bandpass — biquad cascade designed via bilinear transform of a
# 2nd-order Butterworth band-pass. Lightweight and dependency-free (no scipy).
# ---------------------------------------------------------------------------


def _butter_bandpass_biquad(low_hz: float, high_hz: float, fs: int):
    """Compute a single biquad approximation for a bandpass.

    Implemented analytically (RBJ audio-EQ cookbook style) — adequate for
    300–3000 Hz neural-band shaping at fs=1 kHz–30 kHz.
    """
    f0 = np.sqrt(low_hz * high_hz)
    bw = max(high_hz - low_hz, 1.0)
    w0 = 2.0 * np.pi * f0 / fs
    Q = f0 / bw
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    # Normalize by a0.
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


@dataclass
class BandpassState:
    coeffs: tuple  # (b0, b1, b2, a1, a2)
    # Per-channel z-states.
    z1: np.ndarray
    z2: np.ndarray

    @classmethod
    def create(cls, low_hz: float, high_hz: float, fs: int, n_channels: int):
        return cls(
            coeffs=_butter_bandpass_biquad(low_hz, high_hz, fs),
            z1=np.zeros(n_channels, dtype=np.float32),
            z2=np.zeros(n_channels, dtype=np.float32),
        )


def bandpass_apply(state: BandpassState, x: np.ndarray) -> np.ndarray:
    """Apply biquad bandpass to a chunk (T, C).

    Uses the Direct-Form-II Transposed structure for numerical stability.
    """
    b0, b1, b2, a1, a2 = state.coeffs
    z1 = state.z1.copy()
    z2 = state.z2.copy()
    out = np.empty_like(x, dtype=np.float32)
    for t in range(x.shape[0]):
        v = x[t]
        y = b0 * v + z1
        z1 = b1 * v - a1 * y + z2
        z2 = b2 * v - a2 * y
        out[t] = y
    state.z1 = z1
    state.z2 = z2
    return out


# ---------------------------------------------------------------------------
# Threshold spike detection.
# ---------------------------------------------------------------------------


def detect_spikes(
    filtered: np.ndarray,
    threshold_sigma: float = 4.0,
    refractory_samples: int = 8,
) -> np.ndarray:
    """Threshold-cross spike detection.

    Returns an int8 array (T, C) with 1 at the detected spike onset sample.
    Threshold is computed per-channel from a robust noise estimate
    (median absolute deviation), as is standard in neural recording pipelines.
    """
    T, C = filtered.shape
    mad = np.median(np.abs(filtered - np.median(filtered, axis=0)), axis=0)
    sigma = 1.4826 * mad + 1e-6
    threshold = -threshold_sigma * sigma  # negative-going
    out = np.zeros((T, C), dtype=np.int8)
    last_spike = np.full(C, -refractory_samples - 1, dtype=np.int64)
    for t in range(T):
        below = filtered[t] < threshold
        eligible = t - last_spike > refractory_samples
        fire = below & eligible
        if fire.any():
            out[t, fire] = 1
            last_spike[fire] = t
    return out


def extract_waveforms(
    filtered: np.ndarray, spikes: np.ndarray, half_window: int = 16
) -> np.ndarray:
    """Cut spike-triggered waveforms from a filtered signal.

    Returns (n_spikes, 2*half_window+1) waveform snippets, ignoring spikes
    that fall too close to the trace edges.
    """
    out: List[np.ndarray] = []
    T = filtered.shape[0]
    for ch in range(filtered.shape[1]):
        idx = np.flatnonzero(spikes[:, ch])
        for s in idx:
            if s - half_window < 0 or s + half_window + 1 > T:
                continue
            out.append(filtered[s - half_window : s + half_window + 1, ch])
    if not out:
        return np.zeros((0, 2 * half_window + 1), dtype=np.float32)
    return np.stack(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Windowed feature extraction.
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    window_samples: int = 50           # 50 ms at 1 kHz
    hop_samples: int = 25
    spike_threshold_sigma: float = 4.0
    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 200.0


def compute_features(
    signal: np.ndarray,
    cfg: FeatureConfig,
    fs: int,
) -> np.ndarray:
    """Compute per-window features.

    For each non-overlapping (or overlapping, depending on hop) window we
    return:
        * spike count per channel
        * mean band power per channel (RMS of filtered signal)
        * mean absolute value per channel
    Output shape: (n_windows, 3 * n_channels) float32.
    """
    T, C = signal.shape
    bp = BandpassState.create(cfg.bandpass_low_hz, cfg.bandpass_high_hz, fs, C)
    filtered = bandpass_apply(bp, signal)
    spikes = detect_spikes(filtered, cfg.spike_threshold_sigma)

    win = cfg.window_samples
    hop = cfg.hop_samples
    if T < win:
        return np.zeros((0, 3 * C), dtype=np.float32)
    n_win = 1 + (T - win) // hop
    feats = np.empty((n_win, 3 * C), dtype=np.float32)
    for i in range(n_win):
        a = i * hop
        b = a + win
        seg = filtered[a:b]
        sp = spikes[a:b]
        feats[i, :C] = sp.sum(axis=0)
        feats[i, C : 2 * C] = np.sqrt(np.mean(seg * seg, axis=0))
        feats[i, 2 * C :] = np.mean(np.abs(seg), axis=0)
    return feats


def window_labels(
    labels: np.ndarray,
    cfg: FeatureConfig,
) -> np.ndarray:
    """Majority-vote label per window so labels align with features."""
    T = labels.shape[0]
    win = cfg.window_samples
    hop = cfg.hop_samples
    if T < win:
        return np.zeros((0,), dtype=np.int64)
    n_win = 1 + (T - win) // hop
    out = np.empty(n_win, dtype=np.int64)
    for i in range(n_win):
        a = i * hop
        b = a + win
        seg = labels[a:b]
        # Mode via bincount.
        out[i] = int(np.bincount(seg).argmax())
    return out
