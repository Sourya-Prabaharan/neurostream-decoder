"""NeuroStream Decoder — neural signal decoding engine.

Modules:
    signal_gen   – synthetic multi-channel neural recordings
    processing   – streaming-friendly filters, spike detection, features
    models       – GRU + lightweight transformer sequence decoders
    streaming    – continuous window inference runtime
    benchmarks   – throughput + latency percentile reporting
"""

from .device import get_device, device_report

__all__ = ["get_device", "device_report"]
__version__ = "0.1.0"
