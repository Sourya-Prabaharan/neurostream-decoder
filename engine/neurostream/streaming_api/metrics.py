"""Running aggregator for streaming pipeline metrics.

Captures per-window latency components (queue wait, preprocessing,
inference, end-to-end), throughput (windows/sec, packets/sec), and
periodic memory / CPU snapshots. Designed to be O(1) per record by
keeping the raw samples and computing percentiles on demand.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

try:  # optional, used for RSS / CPU
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback
    psutil = None  # type: ignore


@dataclass
class WindowRecord:
    stream_id: str
    seq: int
    t_recv: float       # when packet arrived at consumer
    t_pre_end: float    # preprocessing finished
    t_inf_end: float    # inference finished
    t_send: float       # producer-side timestamp on packet
    queue_wait_ms: float
    preprocess_ms: float
    inference_ms: float
    end_to_end_ms: float
    pred: int
    conf: float


@dataclass
class ThroughputSnapshot:
    t: float
    windows_per_sec: float
    packets_per_sec: float
    rss_mb: float
    cpu_percent: float


class MetricsAggregator:
    def __init__(self):
        self.records: List[WindowRecord] = []
        self.snapshots: List[ThroughputSnapshot] = []
        self._t0 = time.perf_counter()
        self._last_snap_t = self._t0
        self._last_snap_windows = 0
        self._packets = 0
        self._proc = psutil.Process(os.getpid()) if psutil is not None else None

    def record_window(self, rec: WindowRecord) -> None:
        self.records.append(rec)

    def record_packet(self) -> None:
        self._packets += 1

    def snapshot(self, force: bool = False) -> ThroughputSnapshot:
        now = time.perf_counter()
        dt = now - self._last_snap_t
        windows = len(self.records)
        dw = windows - self._last_snap_windows
        wps = dw / dt if dt > 0 else 0.0
        pps = self._packets / (now - self._t0) if now > self._t0 else 0.0
        rss = 0.0
        cpu = 0.0
        if self._proc is not None:
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                cpu = self._proc.cpu_percent(interval=None)
            except Exception:
                pass
        snap = ThroughputSnapshot(t=now - self._t0, windows_per_sec=wps,
                                  packets_per_sec=pps, rss_mb=rss,
                                  cpu_percent=cpu)
        self.snapshots.append(snap)
        self._last_snap_t = now
        self._last_snap_windows = windows
        return snap

    def percentiles(self, attr: str, ps=(50, 95, 99)) -> Dict[str, float]:
        if not self.records:
            return {f"p{p}": 0.0 for p in ps}
        vals = np.asarray([getattr(r, attr) for r in self.records], dtype=np.float64)
        return {f"p{p}": float(np.percentile(vals, p)) for p in ps}

    def summary(self) -> dict:
        n = len(self.records)
        wall = time.perf_counter() - self._t0
        rss_peak = max((s.rss_mb for s in self.snapshots), default=0.0)
        return {
            "n_windows": n,
            "n_packets": self._packets,
            "wall_seconds": wall,
            "windows_per_second": n / wall if wall > 0 else 0.0,
            "packets_per_second": self._packets / wall if wall > 0 else 0.0,
            "end_to_end_ms": self.percentiles("end_to_end_ms"),
            "queue_wait_ms": self.percentiles("queue_wait_ms"),
            "preprocess_ms": self.percentiles("preprocess_ms"),
            "inference_ms": self.percentiles("inference_ms"),
            "rss_mb_peak": rss_peak,
            "cpu_percent_mean": float(np.mean([s.cpu_percent for s in self.snapshots]))
                if self.snapshots else 0.0,
            "snapshots": [s.__dict__ for s in self.snapshots],
        }
