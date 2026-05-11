"""Lightweight EEG window classifier.

A small 1-D CNN that operates directly on band-passed, z-scored windows
of shape (C, T). Used for streaming inference with batch=1 latency in
mind: kept small so CPU inference clears the per-window deadline at
160 Hz × 0.25 s hop = 4 windows/sec/stream with comfortable headroom.

The model is intentionally not pre-trained on a public BCI checkpoint —
this repo trains a baseline on whatever subset of PhysioNet eegmmidb is
downloaded by ``engine.scripts.fetch_eeg`` and reports honest metrics.
If no training data is present, the model is randomly initialised and
metrics are reported as ``trained=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CNNConfig:
    n_channels: int = 64
    window_samples: int = 160
    n_classes: int = 4
    conv1_filters: int = 16
    conv2_filters: int = 32
    kernel: int = 5
    dropout: float = 0.3


class EEGNetSmall(nn.Module):
    """Compact CNN for motor-imagery EEG window classification.

    Forward expects (B, C, T) and emits logits (B, n_classes).
    Parameter count for the default config is ~6 k — small enough to
    keep CPU forward time under 1 ms on a single window.
    """

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        pad = cfg.kernel // 2
        self.conv1 = nn.Conv1d(cfg.n_channels, cfg.conv1_filters,
                               cfg.kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(cfg.conv1_filters)
        self.pool1 = nn.AvgPool1d(4)
        self.conv2 = nn.Conv1d(cfg.conv1_filters, cfg.conv2_filters,
                               cfg.kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(cfg.conv2_filters)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.conv2_filters, cfg.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = torch.relu(self.bn2(self.conv2(h)))
        h = self.pool2(h).squeeze(-1)
        h = self.drop(h)
        return self.fc(h)


def train_classifier(
    model: nn.Module,
    X: np.ndarray,        # (N, C, T)
    y: np.ndarray,        # (N,)
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> dict:
    device = device or torch.device("cpu")
    model.to(device).train()
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    N = X.shape[0]
    history = []
    for ep in range(epochs):
        idx = rng.permutation(N)
        ep_loss = 0.0
        ep_correct = 0
        for s in range(0, N, batch_size):
            b = idx[s : s + batch_size]
            xb = torch.from_numpy(X[b]).float().to(device)
            yb = torch.from_numpy(y[b]).long().to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * b.size
            ep_correct += int((logits.argmax(-1) == yb).sum().item())
        history.append({
            "epoch": ep,
            "loss": ep_loss / N,
            "train_acc": ep_correct / N,
        })
    model.eval()
    return {"history": history, "epochs": epochs}


@torch.no_grad()
def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             device: Optional[torch.device] = None) -> dict:
    device = device or torch.device("cpu")
    model.to(device).eval()
    xb = torch.from_numpy(X).float().to(device)
    yb = torch.from_numpy(y).long().to(device)
    logits = model(xb)
    pred = logits.argmax(-1)
    acc = float((pred == yb).float().mean().item())
    return {"accuracy": acc, "n": int(X.shape[0])}
