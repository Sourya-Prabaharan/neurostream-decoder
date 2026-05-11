# NeuroStream Decoder project pitch

## One-liner

NeuroStream Decoder is a real-time neural signal decoding engine that turns
synthetic multi-channel brain recordings into movement commands or speech-token
predictions with a CUDA-aware PyTorch pipeline and a deployable live dashboard.

## Short pitch

Brain-computer interfaces are not just an ML problem. They are a latency,
streaming, and systems problem: neural recordings arrive continuously, are noisy,
and must be filtered, featurized, and decoded fast enough to control a cursor,
prosthetic, or communication interface. NeuroStream Decoder models that full
path. It simulates electrode recordings, performs streaming signal processing,
detects spike events, extracts windowed features, runs GRU or transformer
decoders, measures per-window latency, and visualizes the whole system in a
research-style dashboard.

The project is intentionally built to run end-to-end on normal hardware while
remaining GPU-ready. Every component runs on CPU for reproducibility, and the
model/inference path automatically moves to CUDA and mixed precision when a
compatible GPU is available.

## What makes it stronger than a typical ML demo

* **Streaming-first architecture**: the decoder consumes fixed-hop windows and
  preserves filter and recurrent state across calls instead of classifying one
  static file after training.
* **Systems metrics**: the dashboard reports latency percentiles, windows per
  second, confidence, spike counts, and device status, not just accuracy.
* **GPU-aware implementation**: PyTorch device selection, CUDA autocast guards,
  batched tensor paths, and a documented CUDA/TensorRT extension plan are built
  into the project.
* **Reproducible demo data**: synthetic sessions are generated from scripts, so
  the dashboard is deterministic and does not depend on private clinical data.
* **Deployable presentation layer**: the Express API and React dashboard make it
  easy to show the system live without asking reviewers to run notebooks.

## Technical components

| Layer | Implementation |
| --- | --- |
| Signal source | Synthetic 24-channel noisy neural recordings, spike trains, labels, and cursor trajectories |
| Processing | Streaming bandpass approximation, MAD-style threshold spike detection, window features |
| Decoding | GRU and lightweight transformer sequence models in PyTorch |
| Runtime | Stateful streaming inference with per-window latency measurement |
| API | Express routes for sessions, trace windows, telemetry, benchmark data, and architecture metadata |
| Dashboard | React, Vite, Tailwind, shadcn/ui, responsive dark-mode scientific interface |
| Verification | Pytest suite, TypeScript checking, production build, benchmark CLI, GitHub Actions CI |

## Best audience framing

For a recruiter, frame this as an AI systems project: real-time inference,
stream processing, GPU-aware model execution, and a polished product interface.

For a neuroscience lab, frame it as a decoding infrastructure prototype: signal
processing, spike/event features, sequence decoding, synthetic sessions, and
hooks for public neural datasets.

For a CUDA/systems interviewer, frame it as a latency optimization project:
windowed processing, batched tensor operations, CPU fallback, CUDA extension
plan, and TensorRT/ONNX roadmap.
