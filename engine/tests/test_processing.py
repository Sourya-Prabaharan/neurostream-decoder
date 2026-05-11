import numpy as np

from engine.neurostream.signal_gen import GeneratorConfig, generate_session
from engine.neurostream.processing import (
    BandpassState,
    FeatureConfig,
    bandpass_apply,
    compute_features,
    detect_spikes,
    extract_waveforms,
    window_labels,
)


def test_bandpass_streaming_equivalence():
    """Filtering the whole signal at once should equal streaming it in chunks."""
    sess = generate_session(GeneratorConfig(n_channels=4, duration_sec=1.0, seed=3))
    sig = sess.signal
    fs = sess.config.sample_rate

    bp_full = BandpassState.create(5.0, 200.0, fs, 4)
    full = bandpass_apply(bp_full, sig)

    bp_chunks = BandpassState.create(5.0, 200.0, fs, 4)
    parts = []
    for i in range(0, sig.shape[0], 137):
        parts.append(bandpass_apply(bp_chunks, sig[i : i + 137]))
    chunked = np.concatenate(parts)

    assert chunked.shape == full.shape
    assert np.allclose(chunked, full, atol=1e-5)


def test_spike_detection_recovers_true_events():
    cfg = GeneratorConfig(n_channels=2, duration_sec=2.0, spike_rate_hz=20.0, seed=7)
    sess = generate_session(cfg)
    bp = BandpassState.create(5.0, 200.0, cfg.sample_rate, cfg.n_channels)
    filtered = bandpass_apply(bp, sess.signal)
    detected = detect_spikes(filtered, threshold_sigma=4.0)
    # Most detected events should be near true spikes (within 5 samples).
    for ch in range(cfg.n_channels):
        true_idx = np.flatnonzero(sess.spikes[:, ch])
        det_idx = np.flatnonzero(detected[:, ch])
        if not len(det_idx):
            continue
        # Each detected spike should have a true spike within 5 samples.
        hits = 0
        for d in det_idx:
            if np.min(np.abs(true_idx - d)) <= 5:
                hits += 1
        precision = hits / len(det_idx)
        assert precision > 0.5, f"precision={precision} on ch{ch}"


def test_feature_shape_and_label_alignment():
    sess = generate_session(GeneratorConfig(n_channels=8, duration_sec=1.0, seed=4))
    cfg = FeatureConfig()
    feats = compute_features(sess.signal, cfg, sess.config.sample_rate)
    labels = window_labels(sess.labels, cfg)
    assert feats.shape[0] == labels.shape[0]
    assert feats.shape[1] == 3 * 8


def test_extract_waveforms_shape():
    sess = generate_session(GeneratorConfig(n_channels=3, duration_sec=1.0, seed=2))
    bp = BandpassState.create(5.0, 200.0, sess.config.sample_rate, 3)
    f = bandpass_apply(bp, sess.signal)
    sp = detect_spikes(f, 4.0)
    wf = extract_waveforms(f, sp, half_window=10)
    if wf.shape[0] > 0:
        assert wf.shape[1] == 21
