import numpy as np
import pytest
from scipy.signal import butter, lfilter

from rtwm.embedder import WatermarkEmbedder

FS = 48_000
KEY = b"\xAA" * 32


@pytest.fixture(scope="session")
def make_host():
    """Factory for low-pass noise with a speech-like spectrum (<~3 kHz)."""

    def _make(seconds: float, seed: int = 0, rms: float = 0.1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        b, a = butter(4, 3_000 / (FS / 2), "low")
        host = lfilter(b, a, rng.standard_normal(int(seconds * FS)))
        return (host / (np.sqrt(np.mean(host**2)) + 1e-12) * rms).astype(np.float32)

    return _make


@pytest.fixture(scope="session")
def marked_10s(make_host):
    """10 s of speech-like noise carrying ~12 watermark frames."""
    return WatermarkEmbedder(KEY).process(make_host(10.0))
