"""Synthetic neural signal generator.

Produces multi-channel extracellular-style recordings with:
  * 1/f background ("pink-ish") + white Gaussian noise
  * Sparse, channel-correlated spike trains driven by latent factors
  * Movement trajectories (2-D cursor) modulated by latent factors
  * Discrete command/token labels emitted on factor transitions

The generator is deterministic given a seed and tuned to look qualitatively
realistic on a dashboard — it is NOT a biophysical simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# Vocabulary used when task="speech"; small, deliberately limited.
SPEECH_TOKENS = ["LEFT", "RIGHT", "UP", "DOWN", "SELECT", "STOP", "YES", "NO"]
MOVEMENT_CLASSES = ["IDLE", "LEFT", "RIGHT", "UP", "DOWN"]


@dataclass
class GeneratorConfig:
    n_channels: int = 32
    sample_rate: int = 1000          # Hz
    duration_sec: float = 30.0
    n_latent: int = 4
    spike_rate_hz: float = 4.0       # average baseline firing rate / channel
    noise_std: float = 0.45          # white-noise sigma on filtered trace
    pink_alpha: float = 0.9          # 1/f roll-off strength
    seed: int = 42
    task: str = "movement"           # "movement" or "speech"


@dataclass
class SyntheticSession:
    config: GeneratorConfig
    time: np.ndarray                 # (T,)
    signal: np.ndarray               # (T, C) float32, post-noise wideband
    spikes: np.ndarray               # (T, C) int8, 1 where a spike onset fired
    latent: np.ndarray               # (T, K) float32, smooth latent factors
    cursor: np.ndarray               # (T, 2) float32, x/y position
    labels: np.ndarray               # (T,)  int64, class index per sample
    label_names: list = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return self.signal.shape[1]

    @property
    def n_samples(self) -> int:
        return self.signal.shape[0]


def _pink_noise(rng: np.random.Generator, n: int, alpha: float = 1.0) -> np.ndarray:
    """Generate 1/f^alpha noise via FFT shaping."""
    white = rng.standard_normal(n)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = np.fft.rfft(white) / (freqs ** (alpha / 2.0))
    out = np.fft.irfft(spectrum, n)
    out -= out.mean()
    std = out.std()
    if std > 0:
        out /= std
    return out.astype(np.float32)


def _make_latent(rng: np.random.Generator, n: int, k: int, fs: int) -> np.ndarray:
    """Smooth multi-dimensional latent factors via low-pass filtered noise."""
    # Cumulative-sum random walk + smoothing.
    raw = rng.standard_normal((n, k)).astype(np.float32) * 0.05
    walk = np.cumsum(raw, axis=0)
    # Simple moving-average smoothing (window ~ 200 ms).
    win = max(2, int(0.2 * fs))
    kernel = np.ones(win, dtype=np.float32) / win
    smoothed = np.empty_like(walk)
    for j in range(k):
        smoothed[:, j] = np.convolve(walk[:, j], kernel, mode="same")
    # Normalize per-channel.
    smoothed -= smoothed.mean(axis=0, keepdims=True)
    s = smoothed.std(axis=0, keepdims=True) + 1e-6
    return smoothed / s


def _label_from_latent(latent: np.ndarray, task: str) -> Tuple[np.ndarray, list]:
    """Map the latent factors to discrete class labels and return label names."""
    if task == "speech":
        names = SPEECH_TOKENS
    else:
        names = MOVEMENT_CLASSES
    # Use sign + dominant factor to pick a class index.
    dom = np.argmax(np.abs(latent), axis=1)
    sign = (latent[np.arange(len(latent)), dom] > 0).astype(np.int64)
    n_classes = len(names)
    # Spread classes across (dom, sign) combinations; "IDLE" / first token when
    # signal energy is small.
    energy = np.linalg.norm(latent, axis=1)
    raw_idx = (dom * 2 + sign) % n_classes
    idle_mask = energy < 0.35
    raw_idx[idle_mask] = 0
    return raw_idx.astype(np.int64), names


def generate_session(cfg: GeneratorConfig | None = None) -> SyntheticSession:
    """Generate one synthetic recording session end-to-end."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.duration_sec * cfg.sample_rate)
    t = np.arange(n, dtype=np.float32) / cfg.sample_rate

    # Latent factors drive both spikes and cursor.
    latent = _make_latent(rng, n, cfg.n_latent, cfg.sample_rate)

    # Cursor: integrate latent[:,0:2] with friction.
    cursor = np.zeros((n, 2), dtype=np.float32)
    vel = np.zeros(2, dtype=np.float32)
    for i in range(1, n):
        accel = latent[i, :2] * 0.05
        vel = 0.98 * vel + accel
        cursor[i] = cursor[i - 1] + vel
    # Re-center & scale to [-1, 1] for display.
    cursor -= cursor.mean(axis=0, keepdims=True)
    mx = np.max(np.abs(cursor)) + 1e-6
    cursor = (cursor / mx).astype(np.float32)

    # Channel tuning curves: each channel projects from latent + bias.
    proj = rng.standard_normal((cfg.n_latent, cfg.n_channels)).astype(np.float32) * 0.6
    bias = rng.uniform(-0.3, 0.3, size=cfg.n_channels).astype(np.float32)
    drive = latent @ proj + bias  # (T, C)

    # Spikes: Poisson with rate modulated by drive.
    base_rate = cfg.spike_rate_hz / cfg.sample_rate
    rate = np.clip(base_rate * np.exp(drive), 0, 0.5)
    spike_draws = rng.random(rate.shape)
    spikes = (spike_draws < rate).astype(np.int8)

    # Wideband signal: pink noise per channel + spike waveforms.
    signal = np.empty((n, cfg.n_channels), dtype=np.float32)
    waveform = _spike_waveform(cfg.sample_rate)
    wlen = len(waveform)
    for ch in range(cfg.n_channels):
        pink = _pink_noise(rng, n, cfg.pink_alpha) * cfg.noise_std
        white = rng.standard_normal(n).astype(np.float32) * (cfg.noise_std * 0.5)
        trace = pink + white
        # Inject spike waveforms (negative-going, neuroscience-style).
        idx = np.flatnonzero(spikes[:, ch])
        for s in idx:
            end = min(n, s + wlen)
            seg = waveform[: end - s]
            trace[s:end] -= seg * rng.uniform(0.8, 1.4)
        signal[:, ch] = trace

    labels, names = _label_from_latent(latent, cfg.task)

    return SyntheticSession(
        config=cfg,
        time=t,
        signal=signal,
        spikes=spikes,
        latent=latent,
        cursor=cursor,
        labels=labels,
        label_names=names,
    )


def _spike_waveform(fs: int) -> np.ndarray:
    """Approximate extracellular action-potential template (1.5 ms)."""
    duration = 0.0015  # 1.5 ms
    n = max(8, int(duration * fs))
    t = np.linspace(0, 1, n, dtype=np.float32)
    # Sharp negative deflection followed by a slower positive rebound.
    wf = np.exp(-((t - 0.25) ** 2) / 0.005) - 0.5 * np.exp(-((t - 0.6) ** 2) / 0.02)
    wf = wf / (np.max(np.abs(wf)) + 1e-6) * 1.5
    return wf.astype(np.float32)
