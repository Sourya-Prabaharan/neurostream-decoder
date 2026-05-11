"""Pure-Python fallbacks for hot kernels that *could* be CUDA-accelerated.

Replace each function with a native cubin / Triton / TensorRT implementation
once a GPU is available. See ``docs/cuda_notes.md`` for the integration
checklist (memory layout, stream semantics, autograd registration).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def has_native_kernels() -> bool:
    """Return True only when a compiled extension is loaded."""
    return False  # native build is not shipped with the sandbox demo


def cuda_threshold_detect(
    x: torch.Tensor,
    threshold: torch.Tensor,
    refractory: int = 8,
) -> torch.Tensor:
    """Threshold spike detection.

    Pure-PyTorch fallback. A bespoke CUDA kernel would:
      1. Tile (T, C) across blocks of channels
      2. Use a per-channel registered counter for refractory
      3. Write spike indices to a global int32 buffer

    The native kernel can be wired in via ``torch.utils.cpp_extension``
    or ``triton.jit`` — see docs/cuda_notes.md.
    """
    # x: (T, C), threshold: (C,)
    below = x < threshold
    T, C = x.shape
    out = torch.zeros_like(below, dtype=torch.int8)
    last = torch.full((C,), -refractory - 1, dtype=torch.long, device=x.device)
    for t in range(T):
        fire = below[t] & ((t - last) > refractory)
        if fire.any():
            out[t, fire] = 1
            last = torch.where(fire, torch.tensor(t, device=x.device), last)
    return out
