"""Device + GPU awareness.

CUDA acceleration is automatic when a GPU is visible to PyTorch; otherwise
we fall back to CPU. Mixed precision (autocast) is enabled only on CUDA.

Extension points for native CUDA kernels and TensorRT integration are stubbed
in ``engine/neurostream/cuda_ext/`` — see ``docs/cuda_notes.md``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, asdict
from typing import Optional

import torch


@dataclass
class DeviceReport:
    device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: Optional[str]
    torch_version: str
    python_version: str
    mixed_precision: bool

    def to_dict(self):
        return asdict(self)


def get_device() -> torch.device:
    """Return the best available torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_report() -> DeviceReport:
    """Snapshot of the current acceleration capability."""
    has_cuda = torch.cuda.is_available()
    return DeviceReport(
        device="cuda" if has_cuda else "cpu",
        cuda_available=has_cuda,
        cuda_device_count=torch.cuda.device_count() if has_cuda else 0,
        cuda_device_name=torch.cuda.get_device_name(0) if has_cuda else None,
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        mixed_precision=has_cuda,  # autocast only safe on CUDA for now
    )


def autocast_ctx(device: torch.device):
    """Return a context manager for mixed-precision when on CUDA, no-op on CPU."""
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=torch.float16)
    # NOTE: torch.cpu.amp.autocast uses bfloat16 and can occasionally produce
    # NaNs on small models; we skip it to keep decoding numerically stable.
    import contextlib

    return contextlib.nullcontext()
