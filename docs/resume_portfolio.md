# Resume and portfolio copy

## Resume bullets

Choose 2-3 depending on the role.

### AI systems / ML engineering

* Built NeuroStream Decoder, a real-time neural signal decoding system that
  streams synthetic 24-channel neural recordings through filtering, spike
  detection, feature extraction, and GRU/transformer inference.
* Implemented a CUDA-aware PyTorch inference path with automatic CPU/GPU device
  selection, mixed-precision guards, latency telemetry, and reproducible
  benchmark scripts.
* Designed and tested a stateful streaming decoder that preserves filter and
  recurrent state across 25 ms inference hops, reporting per-window confidence,
  spike counts, and latency percentiles.

### Systems / CUDA / performance engineering

* Developed a low-latency BCI-style streaming pipeline with windowed neural
  processing, batched tensor inference, CPU fallback, and documented CUDA C++ /
  TensorRT extension points.
* Benchmarked real-time decoding throughput and latency with a Python CLI and
  dashboard telemetry, validating the engine with automated signal-processing,
  model, and streaming-runtime tests.
* Structured the project as a deployable full-stack system with an Express API,
  React dashboard, GitHub Actions CI, and production build workflow.

### Research / neuroscience

* Built an end-to-end neural decoding research prototype for movement and
  speech-token tasks, including synthetic spike-train generation, neural trace
  visualization, and sequence decoder evaluation.
* Added public dataset integration notes for Allen Brain Observatory, OpenNeuro,
  BNCI Horizon 2020, and PhysioNet EEG Motor Imagery datasets.
* Documented limitations around synthetic data, acquisition latency, spike-band
  filtering, and next steps for real ECoG/EEG dataset loaders.

## Portfolio blurb

NeuroStream Decoder is a full-stack AI systems project that simulates a
brain-computer interface decoding pipeline. The engine generates synthetic
multi-channel neural recordings, performs streaming signal filtering and spike
detection, extracts fixed-window features, and decodes intended movement or
speech-token commands with GRU and transformer models. A live React/Express
dashboard visualizes filtered neural traces, spike events, confidence, cursor
trajectory, throughput, and per-window latency.

Unlike a typical notebook-based EEG classifier, this project focuses on the
infrastructure required for real-time neural interfaces: stateful streaming,
GPU-aware inference, benchmarkable latency, reproducible demo data, automated
tests, and a deployable presentation layer.

## GitHub repository tagline

Real-time neural signal decoding engine with streaming spike detection,
CUDA-aware PyTorch inference, latency benchmarks, and a live BCI dashboard.

## Interview sound bites

* “I wanted to build the part of BCI systems that is usually hidden: the
  low-latency streaming decoder and telemetry infrastructure.”
* “The dashboard is not just a frontend mockup. It is backed by generated neural
  sessions and an Express API, and the data can be regenerated from the Python
  engine.”
* “The project is honest about what is synthetic and what is deployable. It runs
  on CPU, automatically uses CUDA when available, and documents the next native
  kernel and TensorRT steps.”
