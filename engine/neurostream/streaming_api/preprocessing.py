"""Per-window EEG preprocessing for the streaming pipeline.

Implements:

* IIR bandpass (reuses ``BandpassState`` from ``processing.py`` so state
  is preserved across windows for online behavior).
* Per-channel z-score normalization computed on the current window.
* Sliding-window framing (handled by the pipeline / ring buffer).
* rFFT-based spectral feature extraction. We aggregate magnitude into
  five EEG bands (delta/theta/alpha/beta/gamma) and concatenate with
  time-domain summary stats. Output dim is documented in
  ``feature_dim``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..processing import BandpassState, bandpass_apply


# Canonical EEG bands in Hz. Chosen to be standard and well-defined for
# the motor imagery dataset (mu/beta on motor cortex sit in 8–30 Hz).
BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 50.0),
)


@dataclass
class PreprocConfig:
    fs: int = 160
    window_samples: int = 160          # 1.0 s window at 160 Hz
    hop_samples: int = 40              # 250 ms hop -> 75% overlap
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 45.0
    n_channels: int = 64

    def feature_dim(self) -> int:
        # mean + std + rms + band powers (5) per channel.
        per_channel = 3 + len(BANDS)
        return per_channel * self.n_channels


class StreamingPreprocessor:
    """Online preprocessor: keeps filter state between windows."""

    def __init__(self, cfg: PreprocConfig):
        self.cfg = cfg
        self.bp = BandpassState.create(
            cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs, cfg.n_channels
        )

    def reset(self) -> None:
        self.bp = BandpassState.create(
            self.cfg.bandpass_low_hz, self.cfg.bandpass_high_hz,
            self.cfg.fs, self.cfg.n_channels,
        )

    def process(self, window: np.ndarray) -> np.ndarray:
        """One window (T, C) -> feature vector (feature_dim,) float32."""
        cfg = self.cfg
        if window.shape != (cfg.window_samples, cfg.n_channels):
            raise ValueError(
                f"expected ({cfg.window_samples}, {cfg.n_channels}), "
                f"got {window.shape}"
            )
        filt = bandpass_apply(self.bp, window.astype(np.float32))
        # Per-channel z-score normalisation on the current window.
        mu = filt.mean(axis=0, keepdims=True)
        sd = filt.std(axis=0, keepdims=True) + 1e-6
        z = (filt - mu) / sd

        # Time-domain summary.
        mean = z.mean(axis=0)
        std = z.std(axis=0)
        rms = np.sqrt(np.mean(z * z, axis=0))

        # rFFT-based band powers.
        spec = np.fft.rfft(z, axis=0)
        psd = (np.abs(spec) ** 2) / cfg.window_samples
        freqs = np.fft.rfftfreq(cfg.window_samples, d=1.0 / cfg.fs)
        bands = np.zeros((len(BANDS), cfg.n_channels), dtype=np.float32)
        for i, (_, lo, hi) in enumerate(BANDS):
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                bands[i] = psd[mask].sum(axis=0)
        # log(1+x) to compress dynamic range — band power can span many decades.
        log_bands = np.log1p(bands)

        feats = np.concatenate(
            [mean, std, rms, log_bands.reshape(-1)],
            axis=0,
        ).astype(np.float32)
        assert feats.shape[0] == cfg.feature_dim(), (feats.shape, cfg.feature_dim())
        return feats

    def feature_dim(self) -> int:
        return self.cfg.feature_dim()
