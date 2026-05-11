# NeuroStream Decoder

Real-time neural signal decoding pipeline: synthetic multi-channel recordings →
streaming bandpass filter → threshold spike detection → window features →
GRU / Tiny-Transformer sequence decoder → live dashboard with latency,
confidence and 2-D cursor visualisation.

Built as a deployable student / research demo. Everything in this repo runs end
to end on CPU. PyTorch automatically uses CUDA when a GPU is visible.

## Why this project matters

Most student BCI demos stop at offline EEG classification. NeuroStream Decoder is
instead built around the systems problem that makes real neural interfaces hard:
turning noisy, high-rate neural streams into useful outputs under tight latency
constraints. The project demonstrates a complete decoding stack with streaming
state, GPU-aware tensor execution, reproducible synthetic sessions, live
telemetry, and a deployable dashboard that can be shown without special
hardware.

For a concise pitch, live walkthrough, and resume bullets, see:

* [docs/project_pitch.md](docs/project_pitch.md)
* [docs/demo_script.md](docs/demo_script.md)
* [docs/resume_portfolio.md](docs/resume_portfolio.md)

```
neurostream-decoder/
├── engine/                    Python neural decoding engine
│   ├── neurostream/           library: signal_gen, processing, models, streaming
│   │   └── cuda_ext/          documented CUDA / Triton extension skeleton
│   ├── scripts/               generate_demo.py, benchmark.py
│   └── tests/                 pytest suite (14 tests)
├── data/                      pre-generated JSON artifacts served by the API
├── server/                    Express API
├── client/                    React + Vite + Tailwind + shadcn dashboard
└── docs/                      architecture.md, cuda_notes.md
```

---

## Architecture

```
   Multi-channel signal (T × C, 1 kHz)
              │
              ▼
   Streaming biquad bandpass (5–200 Hz)         ◄── per-channel z-state preserved
              │
              ▼
   MAD-threshold spike detection (4 σ, refractory)
              │
              ▼
   Window features (50 ms / 25 ms hop)
   spike_count ‖ rms ‖ mav     →     (3·C)
              │
              ▼
   GRU(96, 2 layers)  or  TinyTransformer(96, 2 layers)
              │
              ▼
   Per-window class logits  →  argmax + softmax
              │
              ▼
   Decoded movement / command + latency telemetry
```

A more detailed diagram with code pointers lives in
[docs/architecture.md](docs/architecture.md).

---

## Quickstart

```bash
# 1. Python engine
pip install -r engine/requirements.txt

# 2. Run the tests (14 tests, ~3s on CPU)
pytest engine/tests

# 3. Re-generate demo session data + benchmark (optional — data/ is pre-populated)
python -m engine.scripts.generate_demo

# 4. Frontend / API
npm install
npm run dev        # Express + Vite on :5000

# 5. Production build
npm run build
NODE_ENV=production node dist/index.cjs
```

Open `http://localhost:5000/` for the live dashboard.

### Standalone benchmark

```bash
python -m engine.scripts.benchmark --task movement --kind gru --duration 30
python -m engine.scripts.benchmark --task speech   --kind transformer --duration 20
```

Sample CPU output on the dev sandbox:

```
Streaming windows     : 799
Throughput            : 578.2 windows/sec
Latency  p50 / p95 / p99 : 1.65 / 1.92 / 2.52 ms
```

---

## Training

`engine/scripts/generate_demo.py` trains a fresh decoder for each demo session
(2–6 epochs is plenty for the synthetic data). For longer runs, call
`engine.neurostream.models.train_decoder` directly:

```python
from engine.neurostream.signal_gen import GeneratorConfig, generate_session
from engine.neurostream.processing import FeatureConfig, compute_features, window_labels
from engine.neurostream.models import ModelConfig, build_model, train_decoder
from engine.neurostream.device import get_device
import torch

sess = generate_session(GeneratorConfig(duration_sec=60.0, task="movement", seed=0))
cfg  = FeatureConfig()
X    = compute_features(sess.signal, cfg, sess.config.sample_rate)
y    = window_labels(sess.labels, cfg)

device = get_device()  # cuda if available, otherwise cpu
model  = build_model(ModelConfig(input_dim=X.shape[1], n_classes=len(sess.label_names)))
losses = train_decoder(model, torch.from_numpy(X), torch.from_numpy(y),
                       device=device, epochs=20)
```

---

## Streaming demo

The engine exposes a true streaming runtime where every window is fed in
isolation and the recurrent / filter state is carried forward by hand:

```python
from engine.neurostream.streaming import StreamingDecoder, StreamConfig

decoder = StreamingDecoder(model, X.shape[1], sess.config.n_channels,
                           StreamConfig(window_samples=50, hop_samples=25),
                           device=device)
for a in range(0, sess.signal.shape[0] - 50, 25):
    out = decoder.step(sess.signal[a : a + 50])
    print(out["pred"], out["conf"], f"{out['latency_ms']:.2f}ms")
```

The dashboard reuses the same `predictions` array via the Express API and
re-plays it in the UI off a `requestAnimationFrame` loop.

---

## GPU / CUDA notes

* PyTorch picks the best available device automatically via
  `engine.neurostream.device.get_device`. No code path is GPU-only.
* Mixed-precision (`torch.cuda.amp.autocast`) is enabled in streaming
  inference only when CUDA is available. CPU autocast is intentionally
  skipped to keep behaviour numerically stable on tiny models.
* Hot kernels (spike detection, feature reduction) are written in plain
  PyTorch so they already hit cuBLAS / cuDNN paths on GPU.
* `engine/neurostream/cuda_ext/` is a **documented skeleton** for native
  CUDA C++ / Triton / TensorRT extensions. The sandbox cannot ship a
  compiled extension, so the package falls back to pure PyTorch
  (`has_native_kernels() == False`). See
  [docs/cuda_notes.md](docs/cuda_notes.md) for the integration plan and
  performance expectations.

---

## Public dataset hooks

To run the decoder on real recordings, plug a dataset loader in front of
`compute_features` / `StreamingDecoder`. Suggested sources:

* [Allen Brain Observatory — Neuropixels](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels)
* [OpenNeuro](https://openneuro.org/) (MEG / ECoG / iEEG)
* [BNCI Horizon 2020 EEG competition data](http://bnci-horizon-2020.eu/database/data-sets)
* [PhysioNet EEG Motor Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/)

Loader sketches live in [docs/architecture.md](docs/architecture.md).

---

## Tests

```
pytest engine/tests -v
============================ 14 passed ============================
npm run check
npm run build
```

Coverage:

* `test_signal_gen.py` — shapes, dtypes, determinism, spike rate sanity,
  label vocabulary, cursor normalisation
* `test_processing.py` — bandpass streaming equivalence
  (chunked == whole-signal), spike detection precision, feature & label
  alignment, waveform extraction shape
* `test_models.py` — GRU & Transformer forward-pass shapes, training
  reduces (or does not blow up) cross-entropy loss
* `test_streaming.py` — streaming runtime emits latency for every window;
  `.step()` calls reproduce `.stream()` output exactly (state is preserved
  across windows)

---

## Limitations

* The synthetic generator is qualitatively realistic but is not a
  biophysical simulator. Spike trains are inhomogeneous-Poisson with a
  cosine-tuned drive. Don’t expect SoTA accuracy on real recordings
  without re-tuning.
* Reported latency in the dashboard is the per-window inference time on
  the host the demo was generated on. It does not include I/O or
  hardware acquisition latency.
* The bundled CUDA extension is a skeleton — no compiled `.so` is
  shipped. PyTorch ops still run on GPU via cuBLAS / cuDNN when present.
* The bandpass implementation uses a single biquad section approximation
  (RBJ cookbook). Adequate for the 5–200 Hz local-field band used here;
  swap in a higher-order SOS cascade if you need sharper roll-off for
  300–3000 Hz spike-band processing.

---

## Roadmap

* Per-channel adaptive thresholding from a streaming MAD estimator
* Trained spike-sorting head (e.g. PCA + k-means refinement)
* WebSocket streaming endpoint so the dashboard can drive a live Python
  generator instead of replaying captured predictions
* TensorRT export path (`torch.onnx.export` → `trtexec`) when a GPU host
  is available; see `docs/cuda_notes.md`
* Real-dataset loaders for the four public corpora listed above

---

## Deployment

Frontend + Express API are deployed together. Build, then start the
production server:

```bash
npm run build
NODE_ENV=production node dist/index.cjs
```

This serves the static `dist/public` build alongside the API routes on
port 5000. The same `data/` directory used by `npm run dev` is read by the
production server.

The dashboard uses hash-based routing (`#/`, `#/architecture`,
`#/benchmark`) so it works under deployed iframe constraints out of the
box.

---

## Repository hygiene

This repo includes:

* `.github/workflows/ci.yml` for Python tests, TypeScript checking, and
  production web builds.
* `CONTRIBUTING.md` with a short development checklist.
* `LICENSE` under MIT terms so the project can be shared publicly.
