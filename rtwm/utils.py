"""
Utility helpers: dB conversions, PN generator, filter builders.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import butter
import hashlib, struct

def db_to_lin(db: float) -> float:
    """Convert decibels to linear gain."""
    return 10 ** (db / 20)

def lin_to_db(lin: float) -> float:
    """Convert linear gain to decibels."""
    return 20 * np.log10(max(lin, 1e-12))

def butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    """Return IIR band-pass filter coefficients (b, a)."""
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return b, a

def pseudorandom_chips(seed: int, length: int) -> np.ndarray:
    """Generate ±1 chips from a reproducible seed."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=length, replace=True).astype(np.int8)

def keyed_seed(key: bytes, counter: int) -> int:
    """Return a 32-bit deterministic seed = SipHash-24(key, counter)."""
    h = hashlib.blake2b(struct.pack(">Q", counter), key=key, digest_size=4)
    return int.from_bytes(h.digest(), "big")
