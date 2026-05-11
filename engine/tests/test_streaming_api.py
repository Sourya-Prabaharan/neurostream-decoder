import asyncio

import numpy as np
import pytest
import torch

from engine.neurostream.streaming_api.classifier import CNNConfig, EEGNetSmall
from engine.neurostream.streaming_api.pipeline import (
    PipelineConfig,
    run_concurrent_streams,
    run_single_stream,
)
from engine.neurostream.streaming_api.preprocessing import (
    PreprocConfig,
    StreamingPreprocessor,
)


def test_preprocessor_shape_and_finite():
    cfg = PreprocConfig(fs=160, window_samples=160, hop_samples=40, n_channels=8)
    pre = StreamingPreprocessor(cfg)
    rng = np.random.default_rng(0)
    feats = pre.process(rng.standard_normal((160, 8)).astype(np.float32))
    assert feats.shape == (cfg.feature_dim(),)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_preprocessor_state_persists_across_windows():
    """Sequential calls must update filter state (no reset between windows)."""
    cfg = PreprocConfig(fs=160, window_samples=160, hop_samples=40, n_channels=4)
    pre = StreamingPreprocessor(cfg)
    rng = np.random.default_rng(1)
    w = rng.standard_normal((160, 4)).astype(np.float32)
    f1 = pre.process(w)
    f2 = pre.process(w)
    # State carried forward → second window's filtered signal differs.
    assert not np.allclose(f1, f2)


def test_eegnet_forward_shape():
    m = EEGNetSmall(CNNConfig(n_channels=8, window_samples=160, n_classes=3))
    x = torch.zeros(2, 8, 160)
    out = m(x)
    assert out.shape == (2, 3)


def test_run_single_stream_smoke():
    cfg = PipelineConfig(fs=160, n_channels=8, window_samples=160,
                         hop_samples=40, packet_samples=16,
                         ring_capacity_samples=2000)
    model = EEGNetSmall(CNNConfig(n_channels=8, window_samples=160, n_classes=3))
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((1600, 8)).astype(np.float32)
    summary = asyncio.run(run_single_stream(sig, cfg, model, stream_id="t",
                                            realtime=False, backend="in-memory"))
    assert summary["n_windows"] > 5
    assert summary["end_to_end_ms"]["p50"] >= 0
    assert summary["concurrent_streams"] == 1


def test_run_concurrent_streams_smoke():
    cfg = PipelineConfig(fs=160, n_channels=8, window_samples=160,
                         hop_samples=40, packet_samples=16,
                         ring_capacity_samples=2000)
    model = EEGNetSmall(CNNConfig(n_channels=8, window_samples=160, n_classes=3))
    rng = np.random.default_rng(0)
    sigs = [rng.standard_normal((1600, 8)).astype(np.float32) for _ in range(2)]
    summary = asyncio.run(run_concurrent_streams(sigs, cfg, model,
                                                 realtime=False,
                                                 backend="in-memory"))
    assert summary["concurrent_streams"] == 2
    assert summary["n_windows"] >= 2 * 5
