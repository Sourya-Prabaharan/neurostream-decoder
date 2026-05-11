import time

import numpy as np
import torch

from engine.neurostream.models import ModelConfig, build_model
from engine.neurostream.signal_gen import GeneratorConfig, generate_session
from engine.neurostream.processing import FeatureConfig, compute_features
from engine.neurostream.streaming import StreamConfig, StreamingDecoder


def test_streaming_runs_and_measures_latency():
    sess = generate_session(GeneratorConfig(n_channels=8, duration_sec=2.0, seed=9))
    cfg = FeatureConfig()
    feats = compute_features(sess.signal, cfg, sess.config.sample_rate)
    model = build_model(ModelConfig(input_dim=feats.shape[1], n_classes=len(sess.label_names)))

    stream_cfg = StreamConfig(
        fs=sess.config.sample_rate,
        window_samples=cfg.window_samples,
        hop_samples=cfg.hop_samples,
    )
    decoder = StreamingDecoder(model, feats.shape[1], 8, stream_cfg, device=torch.device("cpu"))

    t0 = time.perf_counter()
    res = decoder.stream(sess.signal)
    wall = time.perf_counter() - t0

    assert len(res.pred_classes) > 0
    assert len(res.latencies_ms) == len(res.pred_classes)
    assert min(res.latencies_ms) > 0
    # CPU streaming should comfortably be faster than realtime here.
    assert wall < sess.config.duration_sec * 3
    assert res.percentile(50) > 0
    assert res.percentile(95) >= res.percentile(50)


def test_streaming_state_persists_across_calls():
    """Two .step() calls in series must equal one stream pass with the same input."""
    sess = generate_session(GeneratorConfig(n_channels=4, duration_sec=1.0, seed=13))
    cfg = FeatureConfig()
    feats = compute_features(sess.signal, cfg, sess.config.sample_rate)
    model = build_model(ModelConfig(input_dim=feats.shape[1], n_classes=len(sess.label_names)))

    sc = StreamConfig(window_samples=cfg.window_samples, hop_samples=cfg.hop_samples)
    a = StreamingDecoder(model, feats.shape[1], 4, sc, device=torch.device("cpu"))
    # Run via stream():
    res_a = a.stream(sess.signal)

    b = StreamingDecoder(model, feats.shape[1], 4, sc, device=torch.device("cpu"))
    out_b = []
    for i in range(0, sess.signal.shape[0] - sc.window_samples + 1, sc.hop_samples):
        out_b.append(b.step(sess.signal[i : i + sc.window_samples])["pred"])
    assert out_b == res_a.pred_classes
