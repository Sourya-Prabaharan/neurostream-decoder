# Contributing

## Development checklist

Before opening a PR or sharing a build:

```bash
pip install -r engine/requirements.txt
npm install
pytest engine/tests -q
npm run check
npm run build
python -m engine.scripts.benchmark --duration 8 --epochs 3
```

## Project conventions

* Keep the Python engine usable without the web dashboard.
* Keep every CUDA or GPU feature optional; CPU must remain the default fallback.
* Do not claim clinical or real-dataset accuracy unless a real dataset loader and
  evaluation split are added.
* Regenerate `data/*.json` with `python -m engine.scripts.generate_demo` after
  changing the signal generator, feature extractor, or decoder output schema.
* Add or update tests when changing `engine/neurostream/`.
* Keep the dashboard free of browser storage APIs such as `localStorage`,
  `sessionStorage`, IndexedDB, and cookies.

## Good first extensions

* Add a PhysioNet EEG Motor Imagery loader.
* Replace the single-section bandpass approximation with an SOS filter bank.
* Add WebSocket streaming from a live Python process.
* Export the GRU or transformer path to ONNX and test TensorRT on a CUDA host.
