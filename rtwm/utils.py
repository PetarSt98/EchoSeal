"""
Shared helpers — band-plan, resampling, filters, PRNG, etc.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, resample_poly
import hashlib, hmac, struct, secrets

# ---------- band helpers ---------------------------------------------------
BAND_PLAN = [
    (4_000, 6_000),          # mid
    (8_000, 10_000),         # upper-mid
    (16_000, 18_000),        # hi-1
    (18_000, 22_000),        # hi-2
]

def choose_band(key: bytes, frame_ctr: int) -> tuple[int, int]:
    """Frequency-hopping sub-band via keyed HMAC."""
    idx = hmac.new(key, struct.pack(">I", frame_ctr), 'sha256').digest()[0] % len(BAND_PLAN)
    return BAND_PLAN[idx]

# ---------- dsp utilities ---------------------------------------------------
def butter_bandpass(lo, hi, fs, *, order=4):
    nyq = 0.5 * fs
    return butter(order, [lo/nyq, hi/nyq], "band")

def resample_to(fs_target: int, audio: np.ndarray, fs_orig: int) -> tuple[np.ndarray, int]:
    if fs_orig == fs_target: return audio, fs_orig
    gcd = np.gcd(fs_orig, fs_target)
    up, down = fs_target // gcd, fs_orig // gcd
    return resample_poly(audio, up, down), fs_target

# ---------- PRNG -----------------------------------------------------------
class StreamPRNG:
    """
    AES-CTR stream generator seeded by XChaCha20 → HKDF derived 128-bit key.
    """
    def __init__(self, master_key: bytes):
        sub = hashlib.blake2s(master_key, digest_size=16, person=b'EchoSealPN').digest()
        from Crypto.Cipher import AES
        self._aes = AES.new(sub, AES.MODE_ECB)

    def bytes(self, counter: int, n: int = 64) -> bytes:
        out = bytearray()
        ctr = counter
        while len(out) < n:
            block = self._aes.encrypt(ctr.to_bytes(16, 'big'))
            out.extend(block)
            ctr += 1
        return bytes(out[:n])

def pn_bits(prng: StreamPRNG, counter: int, n_bits: int) -> np.ndarray:
    data = prng.bytes(counter, (n_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(data, 'u1'))[:n_bits]
