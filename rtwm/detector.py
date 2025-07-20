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
        # y *= 20

        corr = correlate(y, 2*PREAMBLE - 1, mode="valid")
        thresh = 3* np.std(corr)
        MAX_PEAKS = 200
        peaks = np.where(corr > thresh)[0][:MAX_PEAKS]
        print(f"Band {band}: {len(peaks)} peaks detected, max corr: {np.max(corr) if len(corr) > 0 else 0}")
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
            print(f"[LLR] range = [{llr.min():.3f}, {llr.max():.3f}], mean={llr.mean():.3f}, std={llr.std():.3f}")
            blob = polar_dec(llr)
            if blob is None:
                print(f"[RX] ctr={ctr}: polar decode failed")
                continue
            recovered_bits = np.unpackbits(np.frombuffer(blob, dtype="u1"))
            print(f"[RX] bits: {recovered_bits[:16]}... len={len(recovered_bits)}")
            print(f"[RX] ctr={ctr}, blob len={len(blob)}")
            try:
                plain = self.sec.open(blob)
            except Exception:
                print(f"[RX] ctr={ctr} — decrypt failed")
                continue

            print(f"[RX] trying ctr={ctr}, LLR mean={llr.mean():.3f}, std={llr.std():.3f}")

            if not plain.startswith(b"ESAL"):
                print(f"[RX] ctr={ctr} — bad prefix: {plain[:4]}")
                continue

            if int.from_bytes(plain[4:8], "big") != ctr:
                print(f"[RX] Counter mismatch: plain={int.from_bytes(plain[4:8], 'big')} ≠ expected={ctr}")
                continue                                 # counter mismatch
            nonce = plain[8:16]
            if self.session_nonce and nonce == self.session_nonce:
                return True
            elif self.session_nonce is None:
                self.session_nonce = nonce
                return True                          # authentic frame
        return False

    # ------------------------------------------------------------------ helpers
    def _llr(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        bits = self.sec.pn_bits(frame_id, FRAME_LEN)
        sig = (2 * bits - 1).astype(np.float32)

        preamble_len = len(PREAMBLE)
        chips = frame[preamble_len:]
        ref = sig[preamble_len:]

        # Scale by chip energy, not full frame noise
        signal_power = np.mean(ref ** 2)
        noise_power = np.mean((chips - ref) ** 2)
        snr_linear = signal_power / (noise_power + 1e-9)

        # Basic LLR = 2 * symbol * chip / noise_std²
        llr = 2 * chips * ref / (np.std(chips - ref) + 1e-9)

        # Optionally clamp to avoid polar overconfidence
        llr = np.clip(llr, -10.0, 10.0)

        print(f"[LLR] ctr={frame_id}, noise std={np.std(chips - ref):.4f}, llr std={np.std(llr):.4f}")
        return llr



