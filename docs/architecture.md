# Architecture

NeuroStream Decoder is split cleanly into a Python engine and a TypeScript
fullstack dashboard. The two halves communicate exclusively through JSON
artifacts written to `data/` and a small set of Express routes that read
them.

```
┌──────────────────────┐     generate_demo.py       ┌──────────────────┐
│  Python engine       │ ───────────────────────▶   │   data/*.json    │
│  (PyTorch, NumPy)    │                            │  sessions, ...   │
└──────────────────────┘                            └──────────────────┘
                                                            │
                                                            ▼
┌──────────────────────┐         /api/sessions      ┌──────────────────┐
│  Express server      │ ◀───────────────────────── │  React dashboard │
│  server/routes.ts    │ ─── JSON ─────────────────▶│  client/src/...  │
└──────────────────────┘                            └──────────────────┘
```

This shape was chosen on purpose: it keeps the Node runtime free of a
Python child process (brittle in deployed sandboxes), while still letting
us regenerate every byte of demo data from source.

---

## Python engine layout

```
engine/neurostream/
├── __init__.py
├── device.py        torch.device + autocast helpers
├── signal_gen.py    GeneratorConfig, generate_session
├── processing.py    BandpassState, detect_spikes,
│                    extract_waveforms, compute_features, window_labels
├── models.py        ModelConfig, GRUDecoder, TinyTransformer, train_decoder
├── streaming.py     StreamConfig, StreamingDecoder, StreamingResult
├── benchmarks.py    run_benchmark()
└── cuda_ext/        documented native-kernel skeleton (see cuda_notes.md)
```

### Signal generator

`signal_gen.generate_session` returns a `SyntheticSession` dataclass with:

| field    | shape           | dtype  | meaning                                       |
| -------- | --------------- | ------ | --------------------------------------------- |
| `signal` | `(T, C)`        | f32    | wideband multi-channel recording              |
| `spikes` | `(T, C)`        | i8     | ground-truth spike onset indicator            |
| `latent` | `(T, K)`        | f32    | smooth latent factors driving everything      |
| `cursor` | `(T, 2)`        | f32    | 2-D end-effector trajectory in `[-1, 1]`      |
| `labels` | `(T,)`          | i64    | discrete class index per sample               |

Determinism is enforced by passing a `GeneratorConfig.seed` through a
single `np.random.default_rng` instance.

### Streaming filter

The bandpass is a single biquad section with **transposed Direct-Form II**
state. Two persistent `z` registers per channel allow the same kernel to
process a 1-second batch and a 50-ms streaming window with identical
output. The test `test_bandpass_streaming_equivalence` enforces this.

### Spike detection

`detect_spikes` uses a robust MAD-based noise estimate per channel and a
configurable σ multiplier (default 4). An 8-sample refractory window
prevents double-counting a single negative deflection.

### Sequence decoders

Both decoders take `(B, T, F)` features and emit `(B, T, n_classes)`
logits. The GRU additionally returns its hidden state so the streaming
runtime can carry it across windows. The transformer is stateless; it
runs over the buffered window only.

### Streaming runtime

`StreamingDecoder.step(window)` performs the full pipeline on a single
50 ms window in one call:

1. `bandpass_apply` advances per-channel biquad state.
2. The noise buffer (last 1 s) is updated and a fresh MAD estimate is
   computed for the threshold.
3. The window's three feature vectors (`spike_count`, `rms`, `mav`) are
   concatenated and pushed through the decoder as a length-1 sequence.
4. The runtime times itself end-to-end and reports the result.

`test_streaming_state_persists_across_calls` checks that running the same
signal through `.stream(...)` and through a manual loop of `.step(...)`
gives bit-identical class predictions.

---

## Demo data artifacts

`engine/scripts/generate_demo.py` writes the following into `data/`:

| file                             | content                                                              |
| -------------------------------- | -------------------------------------------------------------------- |
| `sessions.json`                  | session manifest + device report                                     |
| `session_<id>.json`              | signal preview (down-sampled), spike raster, cursor, predictions     |
| `benchmark.json`                 | one full train-stream-measure run                                    |
| `architecture.json`              | UI metadata for the pipeline page                                    |

All four are read by the Express layer and served as-is to React. The
React layer drives a `requestAnimationFrame` loop over the predictions
array to make the dashboard feel live.

---

## Express API

All routes are JSON. Hosted at `/api/...`:

| Route                                          | Returns                                          |
| ---------------------------------------------- | ------------------------------------------------ |
| `GET /api/health`                              | `{ ok, device, n_sessions, timestamp }`          |
| `GET /api/sessions`                            | `{ sessions[], device }`                         |
| `GET /api/sessions/:id`                        | full session detail                              |
| `GET /api/sessions/:id/stream?start=&limit=`   | predictions window (simulates live feed)         |
| `GET /api/sessions/:id/trace?start=&span=`     | filtered signal trace for the requested span    |
| `GET /api/sessions/:id/telemetry`              | class distribution + streaming/training stats    |
| `GET /api/benchmark`                           | benchmark.json                                   |
| `GET /api/architecture`                        | architecture.json (pipeline metadata)            |

All responses are cached in-memory (`Map`) after the first read so demo
data is reloaded from disk at most once per process.

---

## Dashboard (`client/src/`)

```
client/src/
├── App.tsx                   wouter hash routing + ThemeProvider
├── pages/
│   ├── dashboard.tsx         Live decode view (oscilloscope, cursor, KPIs)
│   ├── architecture.tsx      Pipeline diagram + dataset hooks
│   └── benchmark.tsx         Benchmark summary
├── components/
│   ├── Layout.tsx            Sidebar + header + device telemetry
│   ├── Oscilloscope.tsx      Multi-channel SVG trace
│   ├── CursorTrail.tsx       2-D cursor trajectory
│   ├── LatencyChart.tsx      Latency sparkline w/ p50/p95 lines
│   └── Logo.tsx              Inline SVG mark
└── lib/
    ├── types.ts              shared types matching the JSON schema
    └── theme.tsx             dark-by-default ThemeProvider
```

The dashboard never persists state (no localStorage / cookies / IndexedDB),
in line with the deployment sandbox rules. All session state lives in
React state and React Query cache.

---

## Loading real datasets

The engine is wired to consume any `(T, C) float32` array at a known
sample rate. To plug a real dataset in:

```python
from engine.neurostream.processing import (
    BandpassState, FeatureConfig, bandpass_apply, compute_features,
)
from engine.neurostream.streaming import StreamingDecoder, StreamConfig

signal, fs, labels = load_my_dataset()           # your loader
cfg   = FeatureConfig(window_samples=int(0.05*fs), hop_samples=int(0.025*fs))
feats = compute_features(signal, cfg, fs)
# ... train as in README.md ...
sd = StreamingDecoder(model, feats.shape[1], signal.shape[1],
                      StreamConfig(fs=fs, window_samples=cfg.window_samples,
                                   hop_samples=cfg.hop_samples), device=device)
```

Per-dataset notes:

* **Allen Neuropixels** — already in NWB format. Use `pynwb` to extract
  the LFP `ElectricalSeries`. Down-sample to 1 kHz for the default
  feature window.
* **PhysioNet EEG Motor Imagery** — load via `mne.io.read_raw_edf`; the
  `T1/T2` annotations map directly to the `LEFT`/`RIGHT` movement labels
  emitted by `signal_gen`.
* **BNCI Horizon 2020** — `.mat` files; `scipy.io.loadmat` works.
* **OpenNeuro** — heterogeneous; use `mne-bids` for the common
  formats.
