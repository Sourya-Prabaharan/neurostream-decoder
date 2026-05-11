# Live demo script

Use this script for a 2-3 minute walkthrough. The goal is to show that the
project is a working system, not just a model notebook.

## Setup before presenting

```bash
pip install -r engine/requirements.txt
npm install
pytest engine/tests -q
npm run build
npm run start
```

Open `http://localhost:5000/`.

## Talk track

### Opening

“This is NeuroStream Decoder, a real-time neural signal decoding project. The
core idea is to simulate what a BCI system has to do in practice: continuously
ingest noisy neural recordings, filter and featurize them, detect spike-like
events, and decode intended movement or speech tokens with low latency.”

### Live decode page

Point to the top controls.

“The dashboard is replaying a generated 24-channel synthetic recording with a
25 ms inference hop. I can switch between movement and speech-token tasks, adjust
the spike threshold, and compare decoder types.”

Point to the KPI cards.

“These cards show the current decoded output, model confidence, measured
inference latency per window, and spike events per window. The point is not just
classification accuracy; it is whether the pipeline can keep up with a live
signal.”

Point to the oscilloscope.

“This trace panel shows six of the filtered channels with spike events overlaid.
The actual engine keeps state across streaming windows, so each step behaves like
a live acquisition loop rather than a one-shot batch classifier.”

Point to cursor/command visualization.

“For movement decoding, the output is shown as a 2-D cursor or command. For a
speech-style task, the same runtime emits token classes.”

### Pipeline page

Navigate to Pipeline.

“This page lays out the architecture: signal stream, streaming filter, spike
detection, feature extraction, sequence decoder, and real-time output. The Python
engine is separate from the UI, so the demo data can be regenerated and the same
runtime can be benchmarked headlessly.”

### Benchmark page

Navigate to Benchmark.

“The benchmark page summarizes throughput and latency. The project runs on CPU
out of the box, but the model path automatically switches to CUDA when a GPU is
available. Native CUDA kernels and TensorRT export are documented as extension
points rather than faked in the sandbox.”

### Close

“The main reason I built it this way is that real BCIs are constrained by
latency and reliability, not only model accuracy. This project shows that I can
build the full systems stack around a neural decoder: data generation, streaming
signal processing, model inference, telemetry, tests, and a deployable
interface.”

## If asked about limitations

Say this directly:

“The bundled data is synthetic, so this is not claiming clinical-grade decoding.
The value is the architecture and performance pipeline. The next step would be
plugging in public EEG/ECoG datasets, adding a higher-order filter bank, and
testing native CUDA kernels or TensorRT on GPU hardware.”

## Quick commands to show credibility

```bash
pytest engine/tests -q
python -m engine.scripts.benchmark --duration 8 --epochs 3
npm run check
npm run build
```
