"""
rtwm.utils
──────────
Shared DSP / crypto helpers: band-plan, keyed frequency hopping, dB helpers,
Butterworth band-pass, polyphase resampling, an AES-CTR PN generator and the
maximal-length synchronisation sequence.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from typing import Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from scipy.signal import butter, resample_poly

# ──────────────────────────────── band-plan ──────────────────────────────
BAND_PLAN: list[Tuple[int, int]] = [
    (4_000, 6_000),    # mid
    (8_000, 10_000),   # upper-mid
    (16_000, 18_000),  # hi-1
    (18_000, 22_000),  # hi-2
]


def choose_band(key: bytes, frame_ctr: int) -> tuple[int, int]:
    """Deterministic (keyed) per-frame band selection via HMAC-SHA256."""
    digest = hmac.new(key, struct.pack(">I", frame_ctr), "sha256").digest()
    return BAND_PLAN[digest[0] % len(BAND_PLAN)]


# ──────────────────────────── dB/linear helpers ──────────────────────────
def db_to_lin(db: float) -> float:
    """dB → linear amplitude."""
    return 10.0 ** (db / 20.0)


def lin_to_db(lin: float) -> float:
    """Linear amplitude → dB (epsilon-guarded against log(0))."""
    return 20.0 * np.log10(lin + 1e-12)


# ───────────────────────────── DSP utilities ─────────────────────────────
def butter_bandpass(lo: float, hi: float, fs: int, *, order: int = 4):
    """IIR coefficients for an order-`order` Butterworth band-pass."""
    nyq = 0.5 * fs
    return butter(order, [lo / nyq, hi / nyq], "band")


def resample_to(
    fs_target: int, audio: np.ndarray, fs_orig: int
) -> tuple[np.ndarray, int]:
    """Integer-ratio polyphase resampling to `fs_target`."""
    if fs_orig == fs_target:
        return audio, fs_orig
    gcd = math.gcd(fs_orig, fs_target)
    up, down = fs_target // gcd, fs_orig // gcd
    return resample_poly(audio, up, down), fs_target


# ────────────────────────────── PN generator ─────────────────────────────
class StreamPRNG:
    """Deterministic AES-CTR pseudo-random byte source used for PN spreading.

    A 128-bit sub-key is derived from the master key with
    ``BLAKE2s(person=b"EchoSeal")``.  Each frame owns a disjoint 2**64-block
    counter region (``counter = frame_ctr << 64``) so frames never share
    keystream, even though they are generated independently.
    """

    _BLOCK = 16  # AES block size in bytes

    def __init__(self, master_key: bytes) -> None:
        self._key = hashlib.blake2s(
            master_key, digest_size=16, person=b"EchoSeal"
        ).digest()

    def bytes(self, frame_ctr: int, n: int = 64) -> bytes:
        """Return `n` deterministic pseudo-random bytes for `frame_ctr`."""
        if n <= 0:
            return b""
        iv = (frame_ctr << 64).to_bytes(self._BLOCK, "big")
        enc = Cipher(algorithms.AES(self._key), modes.CTR(iv)).encryptor()
        return enc.update(b"\x00" * n) + enc.finalize()


def pn_bits(prng: StreamPRNG, frame_ctr: int, n_bits: int) -> np.ndarray:
    """Return `n_bits` PN bits for a frame as a uint8 array of {0, 1}."""
    data = prng.bytes(frame_ctr, (n_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(data, dtype="u1"))[:n_bits]


# ─────────────────────────── synchronisation seq ─────────────────────────
def mseq_63() -> np.ndarray:
    """Length-63 maximal-length sequence from the primitive polynomial x^6+x^5+1.

    Produces the classic 2-level autocorrelation (peak 63, off-peak -1) that
    makes it an excellent preamble for frame synchronisation.
    """
    length = 63
    state = 0b111111  # any non-zero 6-bit seed
    seq = np.zeros(length, dtype=np.uint8)
    for i in range(length):
        seq[i] = state & 1
        feedback = ((state >> 5) ^ (state >> 4)) & 1
        state = ((state << 1) & 0b111111) | feedback
    return seq
