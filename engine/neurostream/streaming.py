"""Streaming inference runtime.

Consumes overlapping windows of raw signal, runs the bandpass / spike
detection / feature extraction pipeline, and feeds features into a
recurrent decoder one step at a time. Records per-window latency for
benchmarking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import torch

from .device import autocast_ctx
from .models import GRUDecoder, TinyTransformer
from .processing import (
    BandpassState,
    FeatureConfig,
    bandpass_apply,
    detect_spikes,
)


@dataclass
class StreamConfig:
    fs: int = 1000
    window_samples: int = 50          # 50 ms window
    hop_samples: int = 25
    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 200.0
    spike_threshold_sigma: float = 4.0


@dataclass
class StreamingResult:
    timestamps: List[float] = field(default_factory=list)  # window center, s
    pred_classes: List[int] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    spike_counts: List[int] = field(default_factory=list)

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(np.asarray(self.latencies_ms), p))

    def to_dict(self):
        return {
            "timestamps": self.timestamps,
            "pred_classes": self.pred_classes,
            "confidences": self.confidences,
            "latencies_ms": self.latencies_ms,
            "spike_counts": self.spike_counts,
            "p50_latency_ms": self.percentile(50),
            "p95_latency_ms": self.percentile(95),
            "p99_latency_ms": self.percentile(99),
        }


class StreamingDecoder:
    """Runs a decoder over a continuous signal stream.

    Keeps persistent biquad and (for GRU) hidden-state buffers across calls so
    that the system behaves identically whether fed in one chunk or many.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feat_dim: int,
        n_channels: int,
        cfg: StreamConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self.bp_state = BandpassState.create(
            cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs, n_channels
        )
        self.hidden: Optional[torch.Tensor] = None
        self.feat_dim = feat_dim
        self.n_channels = n_channels
        # MAD noise estimator running buffer (last second).
        self._noise_window = max(cfg.fs, cfg.window_samples * 4)
        self._noise_buf = np.zeros((0, n_channels), dtype=np.float32)

    def _features_from_window(self, segment: np.ndarray) -> np.ndarray:
        """One window → flat feature vector."""
        filt = bandpass_apply(self.bp_state, segment)
        # Update noise buffer.
        self._noise_buf = np.concatenate([self._noise_buf, filt])[-self._noise_window :]
        # Threshold spike detection using current MAD estimate.
        mad = np.median(np.abs(self._noise_buf - np.median(self._noise_buf, axis=0)), axis=0)
        sigma = 1.4826 * mad + 1e-6
        thresh = -self.cfg.spike_threshold_sigma * sigma
        spikes = (filt < thresh).astype(np.int8)
        spike_count = spikes.sum(axis=0)
        rms = np.sqrt(np.mean(filt * filt, axis=0))
        mav = np.mean(np.abs(filt), axis=0)
        return np.concatenate([spike_count, rms, mav]).astype(np.float32)

    def step(self, segment: np.ndarray) -> dict:
        """Process one window. Returns {pred, conf, latency_ms, spike_count}."""
        assert segment.shape == (self.cfg.window_samples, self.n_channels), (
            f"expected ({self.cfg.window_samples}, {self.n_channels}), got {segment.shape}"
        )
        start = time.perf_counter()
        feats = self._features_from_window(segment)
        x = torch.from_numpy(feats).to(self.device).view(1, 1, -1)

        with torch.no_grad(), autocast_ctx(self.device):
            if isinstance(self.model, GRUDecoder):
                logits, self.hidden = self.model(x, self.hidden)
            else:
                logits, _ = self.model(x)
        logits = logits.float().squeeze(0).squeeze(0)  # (C,)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "pred": int(pred.item()),
            "conf": float(conf.item()),
            "latency_ms": latency_ms,
            "spike_count": int(feats[: self.n_channels].sum()),
            "rms_mean": float(np.mean(feats[self.n_channels : 2 * self.n_channels])),
        }

    def stream(self, signal: np.ndarray) -> StreamingResult:
        """Run the full session as if it were arriving one window at a time."""
        T, C = signal.shape
        assert C == self.n_channels
        out = StreamingResult()
        for a in range(0, T - self.cfg.window_samples + 1, self.cfg.hop_samples):
            seg = signal[a : a + self.cfg.window_samples]
            r = self.step(seg)
            t_center = (a + self.cfg.window_samples / 2) / self.cfg.fs
            out.timestamps.append(float(t_center))
            out.pred_classes.append(r["pred"])
            out.confidences.append(r["conf"])
            out.latencies_ms.append(r["latency_ms"])
            out.spike_counts.append(r["spike_count"])
        return out
