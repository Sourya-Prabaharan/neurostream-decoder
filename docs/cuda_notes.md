# CUDA / GPU acceleration notes

NeuroStream Decoder is GPU-aware end to end. Nothing in the codebase
requires a GPU, but the design admits clean upgrade paths when one is
present.

## What runs on the GPU today

| Component                          | GPU path                                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------------------------- |
| `GRUDecoder`                       | `nn.GRU` is cuDNN-backed on CUDA, including the multi-layer dropout-fused path            |
| `TinyTransformer`                  | `nn.TransformerEncoderLayer` runs on `torch.bmm` + cuBLAS GEMM kernels                    |
| Training (`train_decoder`)         | `AdamW` + cross-entropy run natively on CUDA                                              |
| Streaming inference (`step`)       | autocast fp16 enabled on CUDA; `torch.no_grad` + persistent hidden state                  |
| Threshold spike detection (stub)   | `cuda_ext.cuda_threshold_detect` is a Python fallback today; CUDA implementation sketched |

Device selection happens once via `engine.neurostream.device.get_device`,
which returns `torch.device('cuda' if cuda else 'cpu')`. All callers go
through this helper so adding a new component automatically inherits the
GPU code path.

## Mixed precision

`engine.neurostream.device.autocast_ctx(device)` returns
`torch.cuda.amp.autocast(dtype=torch.float16)` on CUDA and a no-op
context manager on CPU. We do **not** enable CPU autocast because
`torch.cpu.amp.autocast` uses bfloat16, which can produce NaNs on the
small (96-unit) hidden states used here.

## Native extension plan (`engine/neurostream/cuda_ext/`)

The skeleton package shows where a native CUDA / Triton / TensorRT
implementation would plug in. The Python fallback in
`cuda_ext/stubs.py` documents the kernel shape and serves as the
reference implementation.

```
cuda_ext/
├── __init__.py       exports `has_native_kernels`, `cuda_threshold_detect`
└── stubs.py          PyTorch fallback + per-kernel notes
```

### Integration recipe

1. **Spike detection kernel.** Tile the `(T, C)` filtered signal across
   blocks of channels. One thread per channel maintains a per-channel
   refractory counter in a register. Use a global int32 spike-index
   buffer with atomic append (or write a dense int8 mask if memory is
   plentiful). Bind via `torch.utils.cpp_extension.load(name=..., sources=[...])`
   so PyTorch handles ABI compatibility.
2. **Feature reduction kernel.** Fuse the three per-window reductions
   (spike count, RMS, MAV) into a single launch. Each thread handles one
   `(window, channel)` cell. Significant win over three separate
   `tensor.sum / .square().mean().sqrt() / .abs().mean()` passes.
3. **TensorRT export.** After training the GRU model:
   ```bash
   python -m engine.scripts.export_onnx --out model.onnx
   trtexec --onnx=model.onnx --fp16 --saveEngine=decoder.plan
   ```
   Load the engine via `tensorrt.Runtime` at inference time and replace
   `model.forward` with the TensorRT context's `execute_v2`. Expect 1.5–3×
   throughput improvement on Ampere/Ada GPUs for the GRU; less for the
   transformer because its layer sizes are small enough that the kernel
   launch overhead dominates either way.

### Why the skeleton is documented rather than compiled

The sandbox does not ship the CUDA toolkit and is not allowed to
download arbitrary binaries. We therefore include the contract (function
signatures, expected behaviour, fallback) rather than fake compiled
artefacts. The pure-PyTorch fallback already runs on GPU through cuBLAS
when available, so the practical performance gap is small until a real
workload demonstrates a bottleneck.

## Benchmark expectations

Measured on the development sandbox (CPU only):

| Configuration            | windows/sec | p50  | p95  | p99  |
| ------------------------ | ----------: | ---: | ---: | ---: |
| GRU, movement, 24 ch     |        ~580 | 1.65 | 1.92 | 2.52 |
| GRU, speech, 24 ch       |        ~570 | 1.40 | 1.80 | 2.10 |
| Transformer, 24 ch       |        ~420 | 1.55 | 2.05 | 3.10 |

(units: ms unless noted)

A modern GPU (A100, RTX 4090) typically delivers 5–10× throughput on the
GRU path with autocast fp16 enabled, with p50 dropping below 0.5 ms.

## Numerical sanity

* Streaming bandpass output is bit-equal to whole-signal output thanks
  to the persistent biquad state — verified by
  `tests/test_processing.py::test_bandpass_streaming_equivalence`.
* `tests/test_streaming.py::test_streaming_state_persists_across_calls`
  checks that single-window stepping yields the same predictions as a
  bulk stream call.
* All numerical tests run on CPU, so we know the GPU is not the
  reference implementation — it has to match the CPU output.

## Useful references

* [PyTorch CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html)
* [`torch.utils.cpp_extension`](https://pytorch.org/docs/stable/cpp_extension.html)
* [NVIDIA TensorRT — best practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/index.html)
* [Triton language tutorial](https://triton-lang.org/main/getting-started/tutorials/index.html)
