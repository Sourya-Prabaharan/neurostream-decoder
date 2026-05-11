"""Sequence decoders.

Two architectures are implemented:

  * ``GRUDecoder``  – stacked GRU + linear head (default; light + fast)
  * ``TinyTransformer`` – 2-layer encoder for the same task

Both decoders consume features of shape (B, T, F) and emit per-timestep
class logits (B, T, n_classes) so they can be trained with cross-entropy
against window labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    n_classes: int
    hidden_dim: int = 96
    n_layers: int = 2
    dropout: float = 0.1
    kind: str = "gru"  # "gru" | "transformer"


class GRUDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.gru = nn.GRU(
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=cfg.n_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_dim, cfg.n_classes)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ):
        """Return (logits, new_state) so we can run truly streaming."""
        h = torch.relu(self.proj(x))
        y, new_state = self.gru(h, state)
        return self.head(y), new_state


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=4,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.head = nn.Linear(cfg.hidden_dim, cfg.n_classes)

    def forward(self, x: torch.Tensor, state=None):
        h = torch.relu(self.proj(x))
        y = self.enc(h)
        return self.head(y), None  # no recurrent state


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.kind == "transformer":
        return TinyTransformer(cfg)
    return GRUDecoder(cfg)


def train_decoder(
    model: nn.Module,
    features: torch.Tensor,        # (T, F) full session
    labels: torch.Tensor,          # (T,)   long
    *,
    device: torch.device,
    epochs: int = 8,
    chunk_len: int = 128,
    batch_size: int = 8,
    lr: float = 3e-3,
):
    """Train a decoder on a single synthetic session.

    Chunks the session into overlapping windows and runs standard
    cross-entropy. Returns the list of epoch losses.
    """
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    T = features.shape[0]
    # Build all chunks.
    starts = list(range(0, max(1, T - chunk_len + 1), chunk_len // 2))
    if not starts:
        starts = [0]

    history = []
    for ep in range(epochs):
        ep_losses = []
        perm = torch.randperm(len(starts))
        for i in range(0, len(starts), batch_size):
            batch_starts = [starts[j] for j in perm[i : i + batch_size].tolist()]
            xs = torch.stack(
                [features[s : s + chunk_len] for s in batch_starts if s + chunk_len <= T]
            )
            ys = torch.stack(
                [labels[s : s + chunk_len] for s in batch_starts if s + chunk_len <= T]
            )
            if xs.numel() == 0:
                continue
            xs = xs.to(device)
            ys = ys.to(device)
            opt.zero_grad()
            logits, _ = model(xs)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), ys.reshape(-1))
            loss.backward()
            opt.step()
            ep_losses.append(float(loss.item()))
        history.append(sum(ep_losses) / max(1, len(ep_losses)))
    return history
