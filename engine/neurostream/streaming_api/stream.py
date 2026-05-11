"""Stream backends for EEG packets.

Two implementations share the same async API:

* ``RedisStream``  – production path, uses Redis Streams (XADD/XREAD).
* ``InMemoryStream`` – asyncio.Queue fallback for sandboxed environments
  and tests where no Redis server is running.

Each packet carries (stream_id, seq, fs, samples) so multiple concurrent
streams can be multiplexed into the same pipeline.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import numpy as np


@dataclass
class EEGPacket:
    stream_id: str
    seq: int
    fs: int
    samples: np.ndarray  # (T, C) float32
    t_send: float = field(default_factory=time.perf_counter)


class InMemoryStream:
    """asyncio.Queue-backed stream used when Redis is not available."""

    backend = "in-memory"

    def __init__(self, maxsize: int = 1024):
        self._q: asyncio.Queue[EEGPacket] = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def push(self, pkt: EEGPacket) -> None:
        if self._closed:
            raise RuntimeError("stream closed")
        await self._q.put(pkt)

    async def get(self, timeout: float = 1.0) -> Optional[EEGPacket]:
        try:
            return await asyncio.wait_for(self._q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def stream(self) -> AsyncIterator[EEGPacket]:
        while not self._closed:
            pkt = await self.get(timeout=0.5)
            if pkt is not None:
                yield pkt

    def close(self) -> None:
        self._closed = True

    def qsize(self) -> int:
        return self._q.qsize()


class RedisStream:
    """Redis Streams backed transport.

    Uses XADD on push and XREAD with BLOCK on consume. Designed for the
    production path; we instantiate it only if ``redis.asyncio`` imports
    and a server responds to PING. Otherwise callers should fall back to
    InMemoryStream.
    """

    backend = "redis"

    def __init__(self, url: str, key: str = "neurostream:eeg", maxlen: int = 8192):
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError as e:
            raise RuntimeError("redis package not installed") from e
        self._aioredis = aioredis
        self._url = url
        self._key = key
        self._maxlen = maxlen
        self._client = aioredis.from_url(url)
        self._last_id = "$"
        self._closed = False

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def push(self, pkt: EEGPacket) -> None:
        await self._client.xadd(
            self._key,
            {
                "stream_id": pkt.stream_id,
                "seq": str(pkt.seq),
                "fs": str(pkt.fs),
                "shape": f"{pkt.samples.shape[0]}x{pkt.samples.shape[1]}",
                "dtype": str(pkt.samples.dtype),
                "data": pkt.samples.astype(np.float32).tobytes(),
                "t_send": repr(pkt.t_send),
            },
            maxlen=self._maxlen,
            approximate=True,
        )

    async def get(self, timeout: float = 1.0) -> Optional[EEGPacket]:
        block_ms = int(timeout * 1000)
        resp = await self._client.xread({self._key: self._last_id}, block=block_ms, count=1)
        if not resp:
            return None
        _stream, entries = resp[0]
        if not entries:
            return None
        msg_id, fields = entries[0]
        self._last_id = msg_id
        rows, cols = (int(x) for x in fields[b"shape"].decode().split("x"))
        arr = np.frombuffer(fields[b"data"], dtype=np.float32).reshape(rows, cols).copy()
        return EEGPacket(
            stream_id=fields[b"stream_id"].decode(),
            seq=int(fields[b"seq"]),
            fs=int(fields[b"fs"]),
            samples=arr,
            t_send=float(fields[b"t_send"]),
        )

    async def stream(self) -> AsyncIterator[EEGPacket]:
        while not self._closed:
            pkt = await self.get(timeout=0.5)
            if pkt is not None:
                yield pkt

    def close(self) -> None:
        self._closed = True


async def get_stream(prefer_redis: bool = True) -> tuple[object, str]:
    """Return a stream instance plus a string describing what we got.

    Tries Redis first if ``prefer_redis`` is True and ``REDIS_URL`` is set
    (or default localhost). Falls back to InMemoryStream and reports the
    reason so the README/benchmark output stays honest.
    """
    if prefer_redis:
        url = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        try:
            s = RedisStream(url)
            if await s.ping():
                return s, f"redis:{url}"
            return InMemoryStream(), f"in-memory (redis at {url} unreachable)"
        except Exception as e:
            return InMemoryStream(), f"in-memory (redis import/connect failed: {e})"
    return InMemoryStream(), "in-memory (requested)"
