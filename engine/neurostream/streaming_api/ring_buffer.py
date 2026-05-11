"""Fixed-capacity ring buffer for streamed multi-channel EEG packets.

Stores samples in a pre-allocated NumPy array. Writes are O(1) amortized;
sliding-window reads are O(W). The buffer is the bounded queue between
producer (stream ingest) and consumer (preprocessing / inference) — it
absorbs jitter and bounds memory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class RingBufferStats:
    writes: int
    reads: int
    overruns: int  # writes that overwrote unread data
    fill: int     # current number of unread samples
    capacity: int


class RingBuffer:
    def __init__(self, capacity_samples: int, n_channels: int, dtype=np.float32):
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be > 0")
        self._buf = np.zeros((capacity_samples, n_channels), dtype=dtype)
        self._cap = capacity_samples
        self._n_channels = n_channels
        # Total samples ever written / read. Difference is the unread fill.
        self._w = 0
        self._r = 0
        self._overruns = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def n_channels(self) -> int:
        return self._n_channels

    def fill(self) -> int:
        with self._lock:
            return self._w - self._r

    def stats(self) -> RingBufferStats:
        with self._lock:
            return RingBufferStats(
                writes=self._w,
                reads=self._r,
                overruns=self._overruns,
                fill=self._w - self._r,
                capacity=self._cap,
            )

    def write(self, samples: np.ndarray) -> int:
        """Write a chunk of shape (T, C). Returns the number of overruns.

        If the chunk would push the unread fill beyond capacity, oldest
        unread samples are overwritten and the overrun counter increases.
        This is the documented backpressure behavior for high-rate
        producers: drop oldest rather than block.
        """
        if samples.ndim != 2 or samples.shape[1] != self._n_channels:
            raise ValueError(f"expected (T, {self._n_channels}), got {samples.shape}")
        n = samples.shape[0]
        if n == 0:
            return 0
        if n > self._cap:
            samples = samples[-self._cap :]
            n = self._cap
        with self._lock:
            start = self._w % self._cap
            end = start + n
            if end <= self._cap:
                self._buf[start:end] = samples
            else:
                k = self._cap - start
                self._buf[start:] = samples[:k]
                self._buf[: end - self._cap] = samples[k:]
            self._w += n
            unread = self._w - self._r
            overruns = 0
            if unread > self._cap:
                overruns = unread - self._cap
                self._r += overruns
                self._overruns += overruns
            return overruns

    def latest(self, n: int) -> np.ndarray:
        """Read the most recent n samples without advancing the read cursor.

        Used for sliding-window inference where windows overlap.
        """
        if n <= 0:
            return np.zeros((0, self._n_channels), dtype=self._buf.dtype)
        with self._lock:
            available = min(self._w, self._cap)
            n = min(n, available)
            end = self._w % self._cap
            start = (end - n) % self._cap
            if start < end:
                return self._buf[start:end].copy()
            out = np.empty((n, self._n_channels), dtype=self._buf.dtype)
            k = self._cap - start
            out[:k] = self._buf[start:]
            out[k:] = self._buf[:end]
            return out

    def consume(self, n: int) -> np.ndarray:
        """Pop the oldest n unread samples and advance the read cursor."""
        with self._lock:
            unread = self._w - self._r
            n = min(n, unread)
            if n == 0:
                return np.zeros((0, self._n_channels), dtype=self._buf.dtype)
            start = self._r % self._cap
            end = start + n
            if end <= self._cap:
                out = self._buf[start:end].copy()
            else:
                k = self._cap - start
                out = np.empty((n, self._n_channels), dtype=self._buf.dtype)
                out[:k] = self._buf[start:]
                out[k:] = self._buf[: end - self._cap]
            self._r += n
            return out
