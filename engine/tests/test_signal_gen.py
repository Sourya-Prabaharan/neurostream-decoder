import numpy as np

from engine.neurostream.signal_gen import (
    GeneratorConfig,
    generate_session,
    MOVEMENT_CLASSES,
    SPEECH_TOKENS,
)


def test_shapes_and_dtypes():
    cfg = GeneratorConfig(n_channels=8, duration_sec=1.0, seed=0)
    s = generate_session(cfg)
    n = int(cfg.duration_sec * cfg.sample_rate)
    assert s.signal.shape == (n, cfg.n_channels)
    assert s.spikes.shape == (n, cfg.n_channels)
    assert s.labels.shape == (n,)
    assert s.signal.dtype == np.float32
    assert s.spikes.dtype == np.int8
    assert s.cursor.shape == (n, 2)


def test_determinism():
    cfg = GeneratorConfig(seed=123, duration_sec=0.5)
    a = generate_session(cfg)
    b = generate_session(cfg)
    assert np.allclose(a.signal, b.signal)
    assert np.array_equal(a.spikes, b.spikes)
    assert np.array_equal(a.labels, b.labels)


def test_spike_rate_reasonable():
    cfg = GeneratorConfig(n_channels=4, duration_sec=2.0, spike_rate_hz=5.0, seed=42)
    s = generate_session(cfg)
    per_channel_rate = s.spikes.sum(axis=0) / cfg.duration_sec
    # Average firing rate should be in a sane range (Poisson w/ drive expands it).
    assert per_channel_rate.mean() > 1.0
    assert per_channel_rate.mean() < 50.0


def test_task_label_vocab():
    s_mv = generate_session(GeneratorConfig(duration_sec=0.5, task="movement"))
    s_sp = generate_session(GeneratorConfig(duration_sec=0.5, task="speech"))
    assert s_mv.label_names == MOVEMENT_CLASSES
    assert s_sp.label_names == SPEECH_TOKENS
    assert s_mv.labels.max() < len(MOVEMENT_CLASSES)
    assert s_sp.labels.max() < len(SPEECH_TOKENS)


def test_cursor_normalized():
    s = generate_session(GeneratorConfig(duration_sec=1.0))
    assert np.all(s.cursor >= -1.001)
    assert np.all(s.cursor <= 1.001)
