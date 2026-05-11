"""Real-time EEG streaming pipeline.

Modules:
    ring_buffer  – fixed-capacity lock-protected ring buffer for EEG samples
    stream       – pluggable stream backend (in-memory, Redis Streams)
    pipeline     – async ingestion / preprocessing / inference pipeline
    metrics      – running latency / throughput / memory aggregator
    app          – FastAPI app exposing /ingest, /metrics, /healthz
"""

from .ring_buffer import RingBuffer
from .metrics import MetricsAggregator
from .stream import InMemoryStream, RedisStream, EEGPacket, get_stream

__all__ = [
    "RingBuffer",
    "MetricsAggregator",
    "InMemoryStream",
    "RedisStream",
    "EEGPacket",
    "get_stream",
]
