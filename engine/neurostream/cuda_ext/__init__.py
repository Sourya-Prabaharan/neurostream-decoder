"""CUDA / TensorRT extension stubs.

This package is intentionally a documented skeleton rather than a working
native extension. The decoder uses standard PyTorch ops, which already
hit cuBLAS / cuDNN kernels when a GPU is available. Bespoke CUDA kernels
or TensorRT engines plug in here.

Build (when a GPU and CUDA toolkit are available):

    python -m engine.neurostream.cuda_ext.build

That entry point intentionally raises NotImplementedError in this sandbox;
see ``docs/cuda_notes.md`` for the full integration plan.
"""

from .stubs import cuda_threshold_detect, has_native_kernels  # noqa: F401
