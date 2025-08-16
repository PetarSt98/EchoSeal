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
        print(f"[VERIFY] Trying band {hop0} first")
        if self._scan_band_multi_frame(signal, hop0):
            return True
        for band in [b for b in BAND_PLAN if b != hop0]:
            if self._scan_band_multi_frame(signal, band):
                return True
        return False

    # ------------------------------------------------------------------ band scan
    def _scan_band_multi_frame(self, signal: np.ndarray, band) -> bool:
        # 1) Band-pass once
        print(f"[SCAN] Band {band}, signal len: {len(signal)}")
        b, a = butter_bandpass(*band, self.fs_target, order=4)
        y = lfilter(b, a, signal.astype(np.float32, copy=False))

        # 2) Filtered preamble template (zero-state), unit-normalize
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
        from scipy.signal import correlate
        corr = correlate(y, tpl, mode='valid') / e_y  # [-1,1]
        print(f"[SCAN] Correlation shape: {corr.shape}, max: {np.max(corr):.3f}, min: {np.min(corr):.3f}")

        # 4) Adaptive threshold + non-max suppression
        med = float(np.median(corr))
        mad = float(np.median(np.abs(corr - med))) + 1e-12
        thr = med + 4.5 * 1.4826 * mad
        min_distance = FRAME_LEN // 2
        print(f"[SCAN] Threshold: {thr:.3f}, median: {med:.3f}, MAD: {mad:.6f}")
        peaks = []
        for i in range(corr.size):
            if corr[i] < thr:
                continue
            lo = max(0, i - min_distance)
            hi = min(corr.size, i + min_distance + 1)
            if corr[i] >= corr[lo:hi].max():
                peaks.append(i)
        if not peaks:
            k = min(5, corr.size)
            peaks = list(np.argsort(corr)[-k:][::-1])
        print(f"[SCAN] Found {len(peaks)} peaks above threshold")
        if peaks:
            print(f"[SCAN] First 5 peak values: {[corr[p] for p in peaks[:5]]}")
        if not peaks:
            print(f"[SCAN] No peaks above threshold, using top-K fallback")
        # 5) Try decode at candidate starts with *time-based* counter window
        tried = 0
        MAX_TRIES = 1000  # overall budget per band pass (fast!)
        WINDOW_NARROW = 8
        WINDOW_WIDE = 64
        PEAK_LIMIT=40

        for peak_idx in peaks[:PEAK_LIMIT]:
            start = peak_idx
            if start + FRAME_LEN > y.size:
                continue
            frame = y[start:start + FRAME_LEN]

            # Estimate frame counter from time index
            ctr_est = int(round(start / FRAME_LEN))
            cand_ctrs: list[int] = []

            # First: narrow window and only counters that hop to this band
            for ctr in range(max(0, ctr_est - WINDOW_NARROW), ctr_est + WINDOW_NARROW + 1):
                if choose_band(self._band_key, ctr) == band:
                    cand_ctrs.append(ctr)

            # Fallback: widen a bit if nothing matched (should be rare)
            if not cand_ctrs:
                for ctr in range(max(0, ctr_est - WINDOW_WIDE), ctr_est + WINDOW_WIDE + 1):
                    if choose_band(self._band_key, ctr) == band:
                        cand_ctrs.append(ctr)
                        if len(cand_ctrs) >= 16:  # cap work
                            break
            print(f"[SCAN] Peak at {start}, est_ctr={ctr_est}, trying counters: {cand_ctrs[:10]}")
            for ctr in cand_ctrs:
                print(f"  Trying ctr={ctr}")
                if self._try_decode_frame(frame, ctr):
                    print(f"  SUCCESS with ctr={ctr}!")
                    return True
                tried += 1
                if tried >= MAX_TRIES:
                    return False
        return False

    def _try_decode_frame(self, frame: np.ndarray, frame_ctr: int) -> bool:
        """Try to decode a single frame with a specific counter."""
        llr = self._llr(frame, frame_ctr)

        # Quality check
        llr_std = np.std(llr)
        print(f"    LLR std: {llr_std:.3f}")
        if llr_std < 0.3:
            print(f"    Rejected - LLR std too low")
            return False

        blob = polar_dec(llr, list_size=64)
        if blob is None:
            blob = polar_dec(-llr, list_size=64)
            if blob is None:
                print(f"    Polar decode failed")
                return False
        print(f"    Polar decode OK, blob len: {len(blob)}")
        try:
            plain = self.sec.open(blob)
            print(f"    Crypto OK, plain len: {len(plain)}")
        except Exception  as e:
            if len(blob) >= 4 and blob[:4] == b"ESAL":
                plain = blob
                print("    Crypto skipped: payload appears to be PLAINTEXT (legacy mode)")
            else:
                print(f"    Crypto failed: {e}")
                return False

        if not plain.startswith(b"ESAL"):
            print(f"    Wrong magic: {plain[:4].hex()}")
            return False

        embedded_ctr = int.from_bytes(plain[4:8], "big")
        if embedded_ctr != frame_ctr:
            print(f"    Counter mismatch: embedded={embedded_ctr}, expected={frame_ctr}")
            return False

        nonce = plain[8:16]
        if self.session_nonce and nonce == self.session_nonce:
            print(f"    SUCCESS - repeat nonce")
            return True
        elif self.session_nonce is None:
            self.session_nonce = nonce
            print(f"    SUCCESS - new nonce: {nonce.hex()}")
            return True
        print(
            f"    Nonce mismatch: got {nonce.hex()}, expected {self.session_nonce.hex() if self.session_nonce else 'None'}")
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
    def _matched_filter_taps(self, band):
        key = (band[0], band[1], self.fs_target)
        h = self._mf_cache.get(key)
        if h is not None:
            return h

        b, a = butter_bandpass(*band, self.fs_target, order=4)

        # --- build a long enough impulse response to capture IIR memory ---
        M_base = max(len(a), len(b))
        # Heuristic: take the larger of 256 samples or 64×(order+1)
        # (at 48 kHz this is ≳5 ms, enough for a 4th-order BPF tail)
        M = max(256, M_base * 64)

        imp = np.zeros(M, dtype=np.float32);
        imp[0] = 1.0
        g = lfilter(b, a, imp).astype(np.float32)

        # --- truncate by energy: keep 99.9% of the impulse energy ---
        e = g * g
        c = np.cumsum(e)
        total = float(c[-1]) + 1e-20
        idx = int(np.searchsorted(c, 0.999 * total))  # 99.9%
        g = g[:idx + 1] if idx + 1 < g.size else g

        # Matched filter is time-reverse of g
        h = g[::-1]
        # Unit-energy normalize
        h /= (np.sqrt(float(np.sum(h * h))) + 1e-12)

        self._mf_cache[key] = h
        return h

    def _llr(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Produce length-N LLRs for the payload. Steps:
          1) matched-filter with long cached taps,
          2) search an integer chip-phase shift using a sign-invariant metric,
          3) despread with PN at the chosen shift,
          4) robust LLR normalization.
        """
        N = N_DEFAULT

        # --- PN for FULL frame (preamble + payload); we need payload part
        pn_full = self.sec.pn_bits(frame_id, FRAME_LEN)
        pn_payload = pn_full[PRE_L:]
        pn_sy = 2.0 * pn_payload.astype(np.float32) - 1.0  # ±1

        # --- payload segment after preamble
        rx = frame[PRE_L:].astype(np.float32, copy=False)
        n = min(rx.size, pn_sy.size)
        if n <= 0:
            return np.zeros(N, dtype=np.float32)
        rx = rx[:n]
        pn_sy = pn_sy[:n]

        # --- matched filter (long, truncated to ~99.9% energy)
        band = choose_band(self._band_key, frame_id)
        h = self._matched_filter_taps(band)
        mf = np.convolve(rx, h, mode="full").astype(np.float32, copy=False)
        offset = len(h) - 1
        mf_raw = mf[offset:offset + n]  # nominal chip-synchronous slice

        # --- guard region to avoid preamble tail bias
        guard = int(max(16, min(64, len(h) // 8)))
        if guard >= mf_raw.size:
            guard = max(0, mf_raw.size // 4)

        # --- integer shift search (sign-invariant): maximize mean |despread|
        MAX_SHIFT = 24
        best_s = 0
        best_score = -1.0
        for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
            if s >= 0:
                a = mf_raw[s:]
                b = pn_sy[:a.size]
            else:
                a = mf_raw[:mf_raw.size + s]
                b = pn_sy[-s: -s + a.size]
            if a.size <= guard + 8:
                continue
            d = a * b
            score = float(np.mean(np.abs(d[guard:])))
            if score > best_score:
                best_score = score
                best_s = s

        # --- apply the best shift
        if best_s >= 0:
            mf_aligned = mf_raw[best_s:]
            pn_used = pn_sy[:mf_aligned.size]
        else:
            mf_aligned = mf_raw[:mf_raw.size + best_s]
            pn_used = pn_sy[-best_s: -best_s + mf_aligned.size]

        despread = mf_aligned * pn_used

        # --- LLR normalization (use tail to estimate mu/sigma)
        tail = despread[guard:] if despread.size > guard + 8 else despread
        mu = float(np.mean(tail))
        llr_raw = despread - mu

        mad = float(np.median(np.abs(tail - float(np.median(tail))))) + 1e-12
        sigma_mad = 1.4826 * mad
        sigma_std = float(np.std(tail)) + 1e-12
        sigma = max(sigma_mad, sigma_std, 0.1)

        scale = float(np.clip(2.0 / (sigma * sigma), 0.5, 30.0))
        llr = np.clip(llr_raw * scale, -12.0, 12.0).astype(np.float32, copy=False)

        # --- ensure length N
        if llr.size != N:
            out = np.zeros(N, dtype=np.float32)
            m = min(llr.size, N)
            out[:m] = llr[:m]
            llr = out
        return llr

    # def _llr(self, frame: np.ndarray, frame_id: int) -> np.ndarray:
    #     """
    #     Produce length-N LLRs for the payload. Steps:
    #       1) matched-filter with long cached taps,
    #       2) search an integer chip-phase shift using a sign-invariant metric,
    #       3) despread with PN at the chosen shift,
    #       4) robust LLR normalization.
    #     """
    #     N = N_DEFAULT
    #
    #     # --- PN for FULL frame (preamble + payload); we need payload part
    #     pn_full = self.sec.pn_bits(frame_id, FRAME_LEN)
    #     pn_payload = pn_full[PRE_L:]
    #     pn_sy = 2.0 * pn_payload.astype(np.float32) - 1.0  # ±1
    #
    #     # --- payload segment after preamble
    #     rx = frame[PRE_L:].astype(np.float32, copy=False)
    #     n = min(rx.size, pn_sy.size)
    #     if n <= 0:
    #         return np.zeros(N, dtype=np.float32)
    #     rx = rx[:n]
    #     pn_sy = pn_sy[:n]
    #
    #     # --- matched filter (long, truncated to ~99.9% energy)
    #     band = choose_band(self._band_key, frame_id)
    #     h = self._matched_filter_taps(band)
    #     mf = np.convolve(rx, h, mode="full").astype(np.float32, copy=False)
    #     offset = len(h) - 1
    #     mf_raw = mf[offset:offset + n]  # nominal chip-synchronous slice
    #
    #     # --- guard region: early samples still carry preamble tail; don't trust them for stats
    #     # heuristic: ~len(h)/8, clamped to [16, 64]
    #     guard = int(max(16, min(64, len(h) // 8)))
    #     if guard >= mf_raw.size:
    #         guard = max(0, mf_raw.size // 4)
    #
    #     # --- integer offset search using a sign-invariant metric (maximize mean |despread|)
    #     MAX_SHIFT = 24
    #     best_s = 0
    #     best_score = -1.0
    #
    #     for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
    #         if s >= 0:
    #             a = mf_raw[s:]
    #             b = pn_sy[:a.size]
    #         else:
    #             a = mf_raw[:mf_raw.size + s]
    #             b = pn_sy[-s: -s + a.size]
    #         if a.size <= guard + 8:  # need some tail to score
    #             continue
    #
    #         d = a * b  # despread at this shift
    #         score = float(np.mean(np.abs(d[guard:])))  # sign-invariant energy
    #         if score > best_score:
    #             best_score = score
    #             best_s = s
    #
    #     # --- apply best shift
    #     if best_s >= 0:
    #         mf_aligned = mf_raw[best_s:]
    #         pn_used = pn_sy[:mf_aligned.size]
    #     else:
    #         mf_aligned = mf_raw[:mf_raw.size + best_s]
    #         pn_used = pn_sy[-best_s: -best_s + mf_aligned.size]
    #
    #     despread = mf_aligned * pn_used
    #
    #     # --- LLR normalization (use tail region for stable stats)
    #     if despread.size > guard + 8:
    #         tail = despread[guard:]
    #     else:
    #         tail = despread
    #
    #     # center using tail (reduces bias from early transient)
    #     mu = float(np.mean(tail))
    #     llr_raw = despread - mu
    #
    #     # robust sigma
    #     mad = float(np.median(np.abs(tail - float(np.median(tail))))) + 1e-12
    #     sigma_mad = 1.4826 * mad
    #     sigma_std = float(np.std(tail)) + 1e-12
    #     sigma = max(sigma_mad, sigma_std, 0.1)
    #
    #     # AWGN-ish scaling; allow stronger confidence if channel is clean
    #     scale = float(np.clip(2.0 / (sigma * sigma), 0.5, 30.0))
    #     llr = np.clip(llr_raw * scale, -12.0, 12.0).astype(np.float32, copy=False)
    #
    #     # --- ensure length N (zero-pad if trimmed due to shift)
    #     if llr.size != N:
    #         out = np.zeros(N, dtype=np.float32)
    #         m = min(llr.size, N)
    #         out[:m] = llr[:m]
    #         llr = out
    #     return llr



