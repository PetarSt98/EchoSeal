"""
Offline watermark verifier – 3 s snippet ⇒ {True | False}.
"""

from __future__ import annotations
import numpy as np
from scipy.signal import lfilter

from .utils import butter_bandpass, pseudorandom_chips, keyed_seed
from .crypto import AESCipher
from .polar import polar_decode

# --------------------------------------------------------------------------- params
N_BITS = 512          # polar block length  (must match TX)
K_BITS = 344          # information bits
MIN_SEC = 3           # spec: verifier works on ≥3 s clip


class WatermarkDetector:
    """
    Reconstruct one valid watermark frame inside an arbitrary clip.
    Algorithm (simple but effective for MVP):
        1. band-pass 18-22 kHz
        2. despread with candidate PN sequences for frame_ctr ∈ [0, max_try)
        3. majority-vote each bit across repetitions in the clip
        4. polar-decode → AES-decrypt; success ⇒ authentic.
    """

    def __init__(
        self,
        key: bytes,
        *,
        fs: int = 48_000,
        band=(18_000, 22_000),
        max_try: int = 256,
    ) -> None:
        self.fs = fs
        self.lo, self.hi = band
        self._b, self._a = butter_bandpass(self.lo, self.hi, fs)
        self._aes = AESCipher(key)
        self._max_try = max_try
        self._key_bytes = key  # needed for PN seed

    # --------------------------------------------------------------------- public
    def verify(self, audio: np.ndarray) -> bool:
        """
        Return True if *any* candidate frame inside `audio` decrypts OK.
        """
        if audio.size < MIN_SEC * self.fs:
            raise ValueError("need ≥3 s audio for reliable verdict")

        y = lfilter(self._b, self._a, audio.astype(np.float32))
        if not np.any(y):
            return False
        y /= np.max(np.abs(y))

        # pre-compute despread sums for every bit position
        bit_sums = np.zeros(N_BITS, dtype=np.float64)

        # search over candidate frame counters
        for ctr in range(self._max_try):
            pn = pseudorandom_chips(
                keyed_seed(self._key_bytes, ctr), y.size
            ).astype(np.float32)
            prod = y * pn

            # aggregate each repeated chip position (stride = N_BITS)
            for b in range(N_BITS):
                bit_sums[b] = np.sum(prod[b::N_BITS])

            codeword = (bit_sums >= 0).astype(np.uint8)
            payload = polar_decode(codeword, N=N_BITS, K=K_BITS)

            # try AES-GCM decrypt – succeeds only if bits are correct
            try:
                plain = self._aes.decrypt(payload)
            except Exception:
                continue  # wrong bits → next candidate

            # minimal sanity: payload starts with “RTWM”
            if plain.startswith(b"RTWM"):
                return True

        return False
