"""
Offline detector – sync, soft LLR, SCL-list polar, adaptive threshold.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, correlate

from .utils       import BAND_PLAN, butter_bandpass, resample_to, choose_band
from .crypto      import SecureChannel
from .polar_fast  import decode as polar_dec, N_DEFAULT

PREAMBLE      = np.array([1, 0, 1] * 21, dtype=np.uint8)[:63]
FRAME_LEN     = len(PREAMBLE) + N_DEFAULT          # 1087 chips
TIGHT_DELTA   = 3                                  # ±3 quick search
WIDE_DELTA    = 200                                # one-time fallback

class WatermarkDetector:
    """Recover EchoSeal watermark from ≥3 s recording."""
    def __init__(self, key32: bytes, *, fs_target: int = 48_000) -> None:
        self.sec          = SecureChannel(key32)
        self.fs_target    = fs_target
        self.session_nonce: bytes | None = None     # 8-byte anti-replay

    # ------------------------------------------------------------------ API
    def verify(self, audio: np.ndarray, fs_in: int) -> bool:
        signal, _ = resample_to(self.fs_target, audio, fs_in)

        # try predicted hop band first, then remaining bands
        hop0 = choose_band(self.sec.master_key, 0)
        for band in [hop0] + [b for b in BAND_PLAN if b != hop0]:
            if self._scan_band(signal, band):
                return True
        return False

    # ------------------------------------------------------------------ band scan
    def _scan_band(self, signal: np.ndarray, band) -> bool:
        b, a = butter_bandpass(*band, self.fs_target)
        y    = lfilter(b, a, signal.astype(np.float32))
        y   /= np.max(np.abs(y)) + 1e-12

        corr = correlate(y, 2*PREAMBLE - 1, mode="valid")
        thresh = 8 * np.std(corr)
        peaks  = np.where(corr > thresh)[0]

        wide_done = False
        for p in peaks:
            if p + FRAME_LEN > y.size:
                continue
            frame = y[p : p + FRAME_LEN]
            est_ctr = p // FRAME_LEN

            # 1) fast ±3 window
            if self._try_window(frame, est_ctr, TIGHT_DELTA):
                return True

            # 2) one-time wider fallback
            if not wide_done and self._try_window(frame, est_ctr, WIDE_DELTA):
                return True
            wide_done = True
        return False

    # ------------------------------------------------------------------ window search
    def _try_window(self, frame: np.ndarray, ctr0: int, delta: int) -> bool:
        for ctr in range(max(0, ctr0 - delta), ctr0 + delta + 1):
            llr  = self._llr(frame, ctr)
            blob = polar_dec(llr)
            if blob is None:
                continue
            try:
                plain = self.sec.open(blob)
            except Exception:
                continue

            if not plain.startswith(b"ESAL"):
                continue

            if int.from_bytes(plain[4:8], "big") != ctr:
                continue                                 # counter mismatch

            nonce = plain[8:16]
            if self.session_nonce is None:
                self.session_nonce = nonce              # establish session
            if nonce == self.session_nonce:
                return True                             # authentic frame
        return False

    # ------------------------------------------------------------------ helpers
    def _llr(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        bits = self.sec.pn_bits(frame_id, FRAME_LEN)
        sig  = (2*bits - 1).astype(np.float32)
        prod = frame * sig
        noise = np.std(frame)
        return prod[len(PREAMBLE):] / (noise + 1e-12)
