"""
Improved watermark detector with better peak detection and frame handling.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, correlate

from rtwm.utils       import BAND_PLAN, butter_bandpass, resample_to, choose_band
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import decode as polar_dec, N_DEFAULT

def mseq_63() -> np.ndarray:
    reg = np.array([1,1,1,1,1,1], dtype=np.uint8)  # x^6 + x + 1
    out = np.empty(63, dtype=np.uint8)
    for i in range(63):
        out[i] = reg[-1]
        fb = reg[-1] ^ reg[0]
        reg[1:] = reg[:-1]
        reg[0] = fb
    return out

PRE_BITS   = mseq_63()
PRE_L      = PRE_BITS.size
FRAME_LEN  = PRE_L + N_DEFAULT
TIGHT_DELTA   = 3                                  # ±3 quick search
WIDE_DELTA    = 200                                # one-time fallback
EPS = 1e-12

class WatermarkDetector:
    """Recover EchoSeal watermark from ≥3 s recording."""
    def __init__(self, key32: bytes, *, fs_target: int = 48_000) -> None:
        self.sec          = SecureChannel(key32)
        self.fs_target    = fs_target
        self.session_nonce: bytes | None = None     # 8-byte anti-replay
        self._band_key = getattr(self.sec, "band_key", key32)
        self._mf_cache = {}

    # ------------------------------------------------------------------ API
    def verify(self, audio: np.ndarray, fs_in: int) -> bool:
        signal, _ = resample_to(self.fs_target, audio, fs_in)
        hop0 = choose_band(self._band_key, 0)
        if self._scan_band_multi_frame(signal, hop0):
            return True
        for band in [b for b in BAND_PLAN if b != hop0]:
            if self._scan_band_multi_frame(signal, band):
                return True
        return False

    # ------------------------------------------------------------------ band scan
    def _scan_band_multi_frame(self, signal: np.ndarray, band) -> bool:
        # 1) Band-pass once
        b, a = butter_bandpass(*band, self.fs_target, order=4)
        y = lfilter(b, a, signal.astype(np.float32, copy=False))

        # 2) Filter the true preamble (zero-state) and unit-normalize template
        pre_sy = 2.0 * PRE_BITS.astype(np.float32) - 1.0
        tpl = lfilter(b, a, pre_sy)
        tpl_norm = float(np.sqrt(np.sum(tpl * tpl)) + 1e-12)
        tpl = tpl / tpl_norm

        L = tpl.size
        if y.size < L:
            return False

        # 3) Normalized cross-correlation (cosine similarity)
        y2 = y * y
        e_y = np.sqrt(np.convolve(y2, np.ones(L, dtype=np.float32), mode='valid')) + 1e-12
        corr = correlate(y, tpl, mode='valid') / e_y  # in [-1,1]

        # 4) Adaptive threshold via MAD and non-maximum suppression
        med = float(np.median(corr))
        mad = float(np.median(np.abs(corr - med))) + 1e-12
        thr = med + 4.5 * 1.4826 * mad  # ~4.5σ equiv
        min_distance = FRAME_LEN // 2

        peaks = []
        for i in range(corr.size):
            if corr[i] < thr:
                continue
            lo = max(0, i - min_distance)
            hi = min(corr.size, i + min_distance + 1)
            if corr[i] >= corr[lo:hi].max():
                peaks.append(i)

        if not peaks:
            # Fallback: try top-K peaks if none pass thr
            k = min(5, corr.size)
            peaks = list(np.argsort(corr)[-k:][::-1])

        # 5) Try decode at candidate starts (cap work)
        tried = 0
        for peak_idx in peaks[:10]:
            start = peak_idx
            if start + FRAME_LEN > y.size:
                continue
            frame = y[start:start + FRAME_LEN]

            MAX_CTR = 5000
            for test_ctr in range(MAX_CTR):
                if choose_band(self._band_key, test_ctr) != band:
                    continue
                if self._try_decode_frame(frame, test_ctr):
                    return True
                tried += 1
                if tried >= 2000:
                    break
        return False

    def _try_decode_frame(self, frame: np.ndarray, frame_ctr: int) -> bool:
        """Try to decode a single frame with a specific counter."""
        llr = self._llr(frame, frame_ctr)

        # Quality check
        llr_std = np.std(llr)
        if llr_std < 0.3:
            return False

        blob = polar_dec(llr)
        if blob is None:
            return False

        try:
            plain = self.sec.open(blob)
        except:
            return False

        if not plain.startswith(b"ESAL"):
            return False

        embedded_ctr = int.from_bytes(plain[4:8], "big")
        if embedded_ctr != frame_ctr:
            return False

        nonce = plain[8:16]
        if self.session_nonce and nonce == self.session_nonce:
            return True
        elif self.session_nonce is None:
            self.session_nonce = nonce
            return True

        return False

    def verify_raw_frame(self, signal: np.ndarray) -> bool:
        if len(signal) == FRAME_LEN:
            # Try a few likely counters; filter with the corresponding band
            for ctr in range(4):
                band = choose_band(self._band_key, ctr)
                b, a = butter_bandpass(*band, self.fs_target, order=4)
                y = lfilter(b, a, signal.astype(np.float32, copy=False))
                if self._try_decode_frame(y, ctr):
                    return True
        band = choose_band(self._band_key, 0)
        return self._scan_band_multi_frame(signal, band)

    def _scan_band(self, signal: np.ndarray, band, skip_filtering=False) -> bool:
        """Legacy method for compatibility."""
        return self._scan_band_multi_frame(signal, band)

    # ------------------------------------------------------------------ window search
    def _try_window(self, frame: np.ndarray, ctr0: int, delta: int) -> bool:
        """Try different frame counter values within window."""
        for ctr in range(max(0, ctr0 - delta), ctr0 + delta + 1):
            if self._try_decode_frame(frame, ctr):
                return True
        return False

    # ------------------------------------------------------------------ helpers
    def _llr(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Calculate log-likelihood ratios for polar decoder.

        The frame should already be filtered and aligned, starting with preamble.
        """
        # Get PN sequence for this frame
        pn_full = self.sec.pn_bits(frame_id, FRAME_LEN)
        pn_payload = pn_full[PRE_L:]

        # Extract payload part
        payload_received = frame[PRE_L:].copy()

        # Ensure consistent lengths
        min_len = min(len(payload_received), len(pn_payload))
        payload_received = payload_received[:min_len]
        pn_payload = pn_payload[:min_len]

        # Convert PN bits to BPSK symbols: {0,1} → {-1,+1}
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        # Despread: multiply received payload with PN sequence
        despread_symbols = payload_received * pn_symbols

        # Improved LLR calculation
        # Remove DC bias first
        despread_mean = np.mean(despread_symbols)
        despread_centered = despread_symbols - despread_mean

        # Robust noise estimation using median absolute deviation
        despread_abs = np.abs(despread_centered)
        mad = np.median(despread_abs)
        noise_estimate = mad * 1.4826  # Convert MAD to std for Gaussian

        # Fallback if noise estimate is too small
        if noise_estimate < 0.01:
            noise_estimate = np.std(despread_centered)

        # Prevent division by zero
        noise_estimate = max(noise_estimate, 0.1)

        # Scale for LLR - we want reasonable values for the polar decoder
        llr_scale = 2.0 / (noise_estimate ** 2)
        llr_scale = np.clip(llr_scale, 0.5, 20.0)  # Limit scaling

        # Apply scaling
        llr = despread_centered * llr_scale

        # Clip to reasonable range
        llr = np.clip(llr, -10.0, 10.0)

        # Ensure correct length for polar decoder
        if len(llr) != N_DEFAULT:
            llr_full = np.zeros(N_DEFAULT, dtype=np.float32)
            llr_full[:min(len(llr), N_DEFAULT)] = llr[:N_DEFAULT]
            llr = llr_full
        return llr.astype(np.float32, copy=False)