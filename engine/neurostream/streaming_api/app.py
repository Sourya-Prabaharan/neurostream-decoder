"""FastAPI app exposing the streaming pipeline.

Endpoints:

* ``GET  /healthz``  – liveness + backend info + ring fill
* ``POST /ingest``   – push one packet (JSON body) to the active stream
* ``GET  /metrics``  – running summary (percentiles, throughput, RSS)
* ``GET  /windows``  – last N decoded window records

The app starts a single in-process consumer task on startup. It's
intentionally single-stream — for multi-stream benchmarks the harness
uses ``pipeline.run_concurrent_streams`` directly, since spawning many
async tasks in one app is closer to the real-world deployment shape.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = object

from .metrics import MetricsAggregator
from .pipeline import (
    PipelineConfig,
    build_default_model,
    consumer_pipeline,
)
from .stream import EEGPacket, InMemoryStream


class IngestBody(BaseModel):  # type: ignore[misc]
    stream_id: str = "s0"
    seq: int = 0
    fs: int = 160
    samples: List[List[float]] = []  # (T, C)


def create_app(cfg: Optional[PipelineConfig] = None,
               n_classes: int = 4) -> "FastAPI":
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi is not installed")
    cfg = cfg or PipelineConfig()
    model = build_default_model(n_classes, cfg.n_channels, cfg.window_samples)
    metrics = MetricsAggregator()
    state: dict = {"stream": None, "task": None, "started_at": None}

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):
        stream = InMemoryStream(maxsize=4096)
        state["stream"] = stream
        state["started_at"] = time.time()
        state["task"] = asyncio.create_task(
            consumer_pipeline(stream, "s0", cfg, model, metrics, max_idle_s=3600)
        )
        try:
            yield
        finally:
            stream.close()
            state["task"].cancel()
            try:
                await state["task"]
            except Exception:
                pass

    app = FastAPI(title="NeuroStream EEG streaming API", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        stream = state["stream"]
        return {
            "ok": True,
            "backend": getattr(stream, "backend", "unknown"),
            "queue_size": stream.qsize() if hasattr(stream, "qsize") else None,
            "uptime_s": time.time() - (state["started_at"] or time.time()),
            "fs": cfg.fs,
            "n_channels": cfg.n_channels,
            "window_samples": cfg.window_samples,
        }

    @app.post("/ingest")
    async def ingest(body: IngestBody):
        stream = state["stream"]
        if stream is None:
            raise HTTPException(503, "stream not ready")
        arr = np.asarray(body.samples, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != cfg.n_channels:
            raise HTTPException(400,
                f"samples must be (T, {cfg.n_channels}), got {arr.shape}")
        pkt = EEGPacket(stream_id=body.stream_id, seq=body.seq,
                        fs=body.fs, samples=arr)
        await stream.push(pkt)
        return {"accepted": int(arr.shape[0])}

    @app.get("/metrics")
    async def get_metrics():
        metrics.snapshot()
        return metrics.summary()

    @app.get("/windows")
    async def get_windows(limit: int = 50):
        recs = metrics.records[-limit:]
        return {"n": len(recs), "windows": [
            {
                "stream_id": r.stream_id, "seq": r.seq,
                "pred": r.pred, "conf": r.conf,
                "queue_wait_ms": r.queue_wait_ms,
                "preprocess_ms": r.preprocess_ms,
                "inference_ms": r.inference_ms,
                "end_to_end_ms": r.end_to_end_ms,
            } for r in recs
        ]}

    return app


# Module-level app for `uvicorn engine.neurostream.streaming_api.app:app`.
if FastAPI is not None:
    app = create_app()
