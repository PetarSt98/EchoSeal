"""
rtwm.utils
──────────
Shared helpers — band-plan, resampling, filters, PRNG, dB/linear conversions.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
from typing import Tuple

import numpy as np
from scipy.signal import butter, resample_poly

# ──────────────────────────────── band-plan ──────────────────────────────
BAND_PLAN: list[Tuple[int, int]] = [
    (4_000, 6_000),   # mid
    (8_000, 10_000),  # upper-mid
    (16_000, 18_000), # hi-1
    (18_000, 22_000), # hi-2
]


def choose_band(key: bytes, frame_ctr: int) -> tuple[int, int]:
    """
    Deterministic (keyed) frequency-hop selection via HMAC-SHA256.

    Returns a tuple (lo, hi) in Hz taken from BAND_PLAN.
    """
    idx = hmac.new(key, struct.pack(">I", frame_ctr), "sha256").digest()[0] % len(
        BAND_PLAN
    )
    return BAND_PLAN[idx]


# ──────────────────────────── dB/linear helpers ──────────────────────────
def db_to_lin(db: float) -> float:
    """dB → linear (amplitude)"""
    return 10.0 ** (db / 20.0)


def lin_to_db(lin: float) -> float:
    """linear (amplitude) → dB"""
    # small epsilon avoids log(0) on silent chunks
    return 20.0 * np.log10(lin + 1e-12)


# ───────────────────────────── DSP utilities ─────────────────────────────
def butter_bandpass(lo: float, hi: float, fs: int, *, order: int = 4):
    """Return IIR coefficients for an order-`order` Butterworth band-pass."""
    nyq = 0.5 * fs
    return butter(order, [lo / nyq, hi / nyq], "band")


def resample_to(
    fs_target: int, audio: np.ndarray, fs_orig: int
) -> tuple[np.ndarray, int]:
    """Fast integer-ratio polyphase resampling to `fs_target`."""
    if fs_orig == fs_target:
        return audio, fs_orig
    gcd = math.gcd(fs_orig, fs_target)
    up, down = fs_target // gcd, fs_orig // gcd
    return resample_poly(audio, up, down), fs_target


# ────────────────────────────── PN generator ─────────────────────────────
# We prefer PyCryptodome for raw AES-ECB; fall back to cryptography if
# PyCryptodome is not available.
try:
    from Crypto.Cipher import AES as _PCAES  # type: ignore
    # from Cryptodome.Cipher import AES as _PCAES
except ModuleNotFoundError:  # pragma: no cover
    _PCAES = None

if _PCAES is None:
    from cryptography.hazmat.primitives.ciphers import Cipher,algorithms,modes
    from cryptography.hazmat.backends import default_backend


class StreamPRNG:
    """
    Deterministic AES-CTR stream generator.

    • A 128-bit sub-key is derived from the 256-bit master key via BLAKE2s(person=b'EchoSealPN').
    • For frame `n`, we reserve a unique 2⁶⁴-block counter space by left-shifting the
      frame counter:   counter = (frame_ctr << 64) | block_idx
    • Generates cryptographically strong pseudo-random bytes for PN spreading.
    """

    def __init__(self, master_key: bytes):
        sub_key = hashlib.blake2s(master_key, digest_size=16, person=b"EchoSealPN").digest()

        if _PCAES is not None:
            self._aes = _PCAES.new(sub_key, _PCAES.MODE_ECB)

            def _enc(block16: bytes) -> bytes:  # type: ignore
                return self._aes.encrypt(block16)

        else:
            _cipher = Cipher(
                algorithms.AES(sub_key),
                modes.ECB(),
                backend=default_backend(),
            )

            def _enc(block16: bytes) -> bytes:  # type: ignore
                enc = _cipher.encryptor()
                return enc.update(block16) + enc.finalize()

        self._enc_block = _enc  # store backend-agnostic encrypt function

    # ------------------------------------------------------------------ API
    def bytes(self, frame_ctr: int, n: int = 64) -> bytes:
        """Return `n` pseudo-random bytes for the given frame counter."""
        out = bytearray()
        base_ctr = frame_ctr << 64  # allocate unique 64-bit space per frame
        ctr = base_ctr
        while len(out) < n:
            block = self._enc_block(ctr.to_bytes(16, "big"))
            out.extend(block)
            ctr += 1
        return bytes(out[:n])


def pn_bits(prng: StreamPRNG, frame_ctr: int, n_bits: int) -> np.ndarray:
    """
    Convenience helper – returns `n_bits` PN bits for a frame as uint8 array {0,1}.
    """
    data = prng.bytes(frame_ctr, (n_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(data, dtype="u1"))[:n_bits]
