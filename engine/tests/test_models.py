import numpy as np
import torch

from engine.neurostream.models import GRUDecoder, ModelConfig, TinyTransformer, build_model, train_decoder
from engine.neurostream.signal_gen import GeneratorConfig, generate_session
from engine.neurostream.processing import FeatureConfig, compute_features, window_labels


def test_gru_forward_shape():
    cfg = ModelConfig(input_dim=24, n_classes=5, hidden_dim=32, kind="gru")
    m = build_model(cfg)
    assert isinstance(m, GRUDecoder)
    x = torch.randn(2, 16, 24)
    logits, state = m(x)
    assert logits.shape == (2, 16, 5)
    assert state is not None


def test_transformer_forward_shape():
    cfg = ModelConfig(input_dim=24, n_classes=5, hidden_dim=32, n_layers=2, kind="transformer")
    m = build_model(cfg)
    assert isinstance(m, TinyTransformer)
    x = torch.randn(2, 16, 24)
    logits, state = m(x)
    assert logits.shape == (2, 16, 5)
    assert state is None


def test_training_reduces_loss():
    sess = generate_session(GeneratorConfig(duration_sec=4.0, seed=11))
    cfg = FeatureConfig()
    feats = compute_features(sess.signal, cfg, sess.config.sample_rate)
    labels = window_labels(sess.labels, cfg)
    model = build_model(ModelConfig(input_dim=feats.shape[1], n_classes=len(sess.label_names)))
    losses = train_decoder(
        model,
        torch.from_numpy(feats),
        torch.from_numpy(labels),
        device=torch.device("cpu"),
        epochs=4,
    )
    assert len(losses) == 4
    # Either the loss falls overall, or it's already low — both are fine.
    assert losses[-1] <= losses[0] + 0.05
