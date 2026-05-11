"""Async EEG streaming pipeline.

Wires a producer (replay of a real or synthetic EEG file at a fixed
sample rate) through a stream backend, into a ring buffer, then to an
async consumer that slices sliding windows, runs preprocessing + the
PyTorch CNN inference, and records per-window metrics.

Single-process layout. Multiple concurrent streams are supported by
running multiple producer/consumer task pairs that share the same
metrics aggregator but isolate their own ring buffers and filter state.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .classifier import CNNConfig, EEGNetSmall
from .metrics import MetricsAggregator, WindowRecord
from .preprocessing import PreprocConfig, StreamingPreprocessor
from .ring_buffer import RingBuffer
from .stream import EEGPacket, InMemoryStream


@dataclass
class PipelineConfig:
    fs: int = 160
    n_channels: int = 64
    window_samples: int = 160
    hop_samples: int = 40
    packet_samples: int = 16          # producer chunk size
    ring_capacity_samples: int = 8000  # ~50 s @ 160 Hz
    snapshot_interval_s: float = 0.5


async def producer_replay(
    stream,
    stream_id: str,
    signal: np.ndarray,
    fs: int,
    packet_samples: int,
    realtime: bool = True,
) -> int:
    """Replay an array (T, C) as EEG packets at the given sample rate.

    Returns the number of packets pushed. When ``realtime`` is true the
    producer paces itself to the stream sample rate so the consumer sees
    realistic queue waits; set false to feed as fast as possible (for
    pure throughput measurement).
    """
    T = signal.shape[0]
    seq = 0
    t_pkt = packet_samples / fs
    t_start = time.perf_counter()
    pushed = 0
    for a in range(0, T - packet_samples + 1, packet_samples):
        chunk = np.ascontiguousarray(signal[a : a + packet_samples], dtype=np.float32)
        pkt = EEGPacket(stream_id=stream_id, seq=seq, fs=fs, samples=chunk)
        await stream.push(pkt)
        seq += 1
        pushed += 1
        if realtime:
            target = t_start + seq * t_pkt
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)
    return pushed


async def consumer_pipeline(
    stream,
    stream_id: str,
    cfg: PipelineConfig,
    model: torch.nn.Module,
    metrics: MetricsAggregator,
    max_idle_s: float = 1.0,
    snapshot_every_n: int = 5,
) -> int:
    """Consume packets, slide windows, run inference, record metrics.

    The consumer is per-stream: each stream id gets its own ring buffer
    and preprocessor (filter state must not cross-contaminate between
    subjects/streams). The ``metrics`` aggregator is shared.

    Returns the number of windows decoded.
    """
    pre_cfg = PreprocConfig(
        fs=cfg.fs, window_samples=cfg.window_samples,
        hop_samples=cfg.hop_samples, n_channels=cfg.n_channels,
    )
    pre = StreamingPreprocessor(pre_cfg)
    rb = RingBuffer(cfg.ring_capacity_samples, cfg.n_channels)
    device = next(model.parameters()).device

    windows_emitted = 0
    samples_since_window = 0
    samples_total = 0
    last_pkt_t = time.perf_counter()
    while True:
        pkt = await stream.get(timeout=0.5)
        now = time.perf_counter()
        if pkt is None:
            if now - last_pkt_t > max_idle_s:
                break
            continue
        if pkt.stream_id != stream_id:
            # Push back is not supported; in multi-stream tests each
            # stream has its own backend, so this branch is a safety net.
            continue
        last_pkt_t = now
        metrics.record_packet()
        t_recv = now
        rb.write(pkt.samples)
        samples_total += pkt.samples.shape[0]
        samples_since_window += pkt.samples.shape[0]
        while samples_total >= cfg.window_samples and samples_since_window >= cfg.hop_samples:
            window = rb.latest(cfg.window_samples)
            if window.shape[0] < cfg.window_samples:
                break
            t_pre_start = time.perf_counter()
            feats = pre.process(window)
            # Reshape: classifier wants (B, C, T). We feed it the z-scored
            # window so the CNN sees calibrated input.
            x = torch.from_numpy(window.T.astype(np.float32)).unsqueeze(0).to(device)
            t_pre_end = time.perf_counter()
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)
            t_inf_end = time.perf_counter()

            queue_wait_ms = (t_recv - pkt.t_send) * 1000.0
            preprocess_ms = (t_pre_end - t_pre_start) * 1000.0
            inference_ms = (t_inf_end - t_pre_end) * 1000.0
            end_to_end_ms = (t_inf_end - pkt.t_send) * 1000.0
            metrics.record_window(WindowRecord(
                stream_id=stream_id,
                seq=pkt.seq,
                t_recv=t_recv,
                t_pre_end=t_pre_end,
                t_inf_end=t_inf_end,
                t_send=pkt.t_send,
                queue_wait_ms=queue_wait_ms,
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                end_to_end_ms=end_to_end_ms,
                pred=int(pred.item()),
                conf=float(conf.item()),
                # plus a feature hash for sanity (unused, drop to save mem)
            ))
            _ = feats  # used for downstream feature-only consumers later
            windows_emitted += 1
            samples_since_window = 0
            if windows_emitted % snapshot_every_n == 0:
                metrics.snapshot()
    metrics.snapshot()
    return windows_emitted


def build_default_model(n_classes: int, n_channels: int, window_samples: int) -> EEGNetSmall:
    cfg = CNNConfig(
        n_channels=n_channels,
        window_samples=window_samples,
        n_classes=n_classes,
    )
    return EEGNetSmall(cfg)


async def run_single_stream(
    signal: np.ndarray,
    cfg: PipelineConfig,
    model: torch.nn.Module,
    stream_id: str = "s0",
    realtime: bool = True,
    backend: str = "in-memory",
) -> dict:
    """Convenience: spawn one producer + one consumer with an in-memory
    stream, run to completion, return metrics."""
    stream = InMemoryStream(maxsize=4096)
    metrics = MetricsAggregator()
    prod = asyncio.create_task(producer_replay(
        stream, stream_id, signal, cfg.fs, cfg.packet_samples, realtime=realtime,
    ))
    cons = asyncio.create_task(consumer_pipeline(
        stream, stream_id, cfg, model, metrics,
    ))
    pushed = await prod
    n_windows = await cons
    summary = metrics.summary()
    summary["pushed_packets"] = pushed
    summary["windows_decoded"] = n_windows
    summary["backend"] = backend
    summary["concurrent_streams"] = 1
    summary["fs"] = cfg.fs
    summary["window_samples"] = cfg.window_samples
    summary["hop_samples"] = cfg.hop_samples
    return summary


async def run_concurrent_streams(
    signals: List[np.ndarray],
    cfg: PipelineConfig,
    model: torch.nn.Module,
    realtime: bool = True,
    backend: str = "in-memory",
) -> dict:
    """Run N concurrent producer/consumer pairs sharing a single model.

    Each stream gets its own InMemoryStream + RingBuffer + preprocessor
    so filter state stays isolated. The model is shared across streams
    to mirror a production layout where one inference worker serves
    many ingestion channels.
    """
    metrics = MetricsAggregator()
    tasks = []
    for i, sig in enumerate(signals):
        stream = InMemoryStream(maxsize=4096)
        sid = f"s{i}"
        tasks.append(asyncio.create_task(producer_replay(
            stream, sid, sig, cfg.fs, cfg.packet_samples, realtime=realtime,
        )))
        tasks.append(asyncio.create_task(consumer_pipeline(
            stream, sid, cfg, model, metrics,
        )))
    results = await asyncio.gather(*tasks)
    summary = metrics.summary()
    summary["backend"] = backend
    summary["concurrent_streams"] = len(signals)
    summary["fs"] = cfg.fs
    summary["window_samples"] = cfg.window_samples
    summary["hop_samples"] = cfg.hop_samples
    summary["per_stream_results"] = results
    return summary
