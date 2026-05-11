import numpy as np
import pytest

from engine.neurostream.streaming_api.ring_buffer import RingBuffer


def test_write_and_consume_round_trip():
    rb = RingBuffer(capacity_samples=100, n_channels=4)
    chunk = np.arange(40 * 4, dtype=np.float32).reshape(40, 4)
    overruns = rb.write(chunk)
    assert overruns == 0
    assert rb.fill() == 40
    out = rb.consume(40)
    np.testing.assert_array_equal(out, chunk)
    assert rb.fill() == 0


def test_overrun_drops_oldest():
    rb = RingBuffer(capacity_samples=10, n_channels=2)
    rb.write(np.zeros((10, 2), dtype=np.float32))
    overruns = rb.write(np.ones((5, 2), dtype=np.float32))
    # 5 oldest unread samples should have been dropped.
    assert overruns == 5
    assert rb.fill() == 10
    s = rb.stats()
    assert s.overruns == 5


def test_latest_returns_most_recent_without_advancing():
    rb = RingBuffer(capacity_samples=8, n_channels=1)
    rb.write(np.arange(8, dtype=np.float32).reshape(8, 1))
    last3 = rb.latest(3)
    np.testing.assert_array_equal(last3.ravel(), [5, 6, 7])
    # Read cursor unchanged.
    assert rb.fill() == 8


def test_wraparound_correctness():
    rb = RingBuffer(capacity_samples=6, n_channels=1)
    rb.write(np.arange(4, dtype=np.float32).reshape(4, 1))
    rb.consume(2)  # advance read cursor to 2
    rb.write(np.arange(10, 14, dtype=np.float32).reshape(4, 1))
    # latest(4) should be [10, 11, 12, 13]
    np.testing.assert_array_equal(rb.latest(4).ravel(), [10, 11, 12, 13])


def test_write_shape_validation():
    rb = RingBuffer(capacity_samples=8, n_channels=3)
    with pytest.raises(ValueError):
        rb.write(np.zeros((4, 2), dtype=np.float32))
