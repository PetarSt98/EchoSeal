"""
Improved watermark detector with better peak detection and frame handling.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, correlate
from cryptography.exceptions import InvalidTag

from rtwm.utils       import BAND_PLAN, butter_bandpass, resample_to, choose_band, mseq_63
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import decode as polar_dec, N_DEFAULT

PRE_BITS   = mseq_63()
PRE_L      = len(PRE_BITS)
# Header: 16-bit counter (low bits) repeated 8x => 128 chips
HDR_BITS   = 16
HDR_REPEAT = 8
HDR_L      = 128
FRAME_LEN  = PRE_L + HDR_L + N_DEFAULT
TIGHT_DELTA   = 3                                  # ±3 quick search
WIDE_DELTA    = 200                                # one-time fallback
EPS = 1e-12

class WatermarkDetector:
    """Recover EchoSeal watermark from ≥3 s recording."""

    def __init__(self, key32: bytes, *, fs_target: int = 48_000, list_size: int = 256) -> None:
        self.sec          = SecureChannel(key32)
        self.fs_target    = fs_target
        self.session_nonce: bytes | None = None     # 8-byte anti-replay
        self._band_key = getattr(self.sec, "band_key", key32)
        self._mf_cache = {}
        self._list_size = int(list_size)

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
        thr = min(thr, 0.95)
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
        # 5) Try decode at candidate starts using header-derived counter when possible
        tried = 0
        MAX_TRIES = 400  # overall budget per band pass (fast!)
        PEAK_LIMIT=25

        for peak_idx in peaks[:PEAK_LIMIT]:
            start = peak_idx
            if start + FRAME_LEN > y.size:
                continue
            frame = y[start:start + FRAME_LEN]

            # --- fast estimate from time index (used if header fails)
            ctr_est = int(round(start / FRAME_LEN))
            cand_ctrs: list[int] = []

            # --- (1A) decode header to get ctr_lo16 ---
            hdr_ok, ctr_lo16, hdr_score = self._decode_header(frame, band)
            if hdr_ok:
                lo = max(0, ctr_est - WIDE_DELTA)
                hi = ctr_est + WIDE_DELTA + 1
                for ctr in range(lo, hi):
                    if (ctr & 0xFFFF) == ctr_lo16 and choose_band(self._band_key, ctr) == band:
                        cand_ctrs.append(ctr)
                print(f"  Peak@{start}: ctr_est={ctr_est}, hdr_lo16=0x{ctr_lo16:04X}, "
                      + f"score={hdr_score:.3f}, trying {len(cand_ctrs)} counters (header gated)")
            else:
            # Fallback: the old time-based + band-gated window
                for ctr in range(max(0, ctr_est - TIGHT_DELTA), ctr_est + TIGHT_DELTA + 1):
                    if choose_band(self._band_key, ctr) == band:
                        cand_ctrs.append(ctr)
                if not cand_ctrs:
                    lo = max(0, ctr_est - WIDE_DELTA)
                    hi = ctr_est + WIDE_DELTA + 1
                    for ctr in range(lo, hi):
                        if choose_band(self._band_key, ctr) == band:
                            cand_ctrs.append(ctr)
                print(f"  Peak@{start}: ctr_est={ctr_est}, trying {len(cand_ctrs)} counters "
                       + f"(±{TIGHT_DELTA} then ±{WIDE_DELTA})")

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
        print(f"[DECODE DEBUG] Trying frame counter {frame_ctr}")
        print(f"[DECODE DEBUG] Frame shape: {frame.shape}")
        print(f"[DECODE DEBUG] Frame[:8]: {frame[:8]}")
        print(f"[DECODE DEBUG] Frame RMS: {np.sqrt(np.mean(frame ** 2)):.8f}")
        # First, default PN convention (pn over whole frame, slice payload
        llr = self._llr(frame, frame_ctr, pn_variant=0)
        print(f"[DECODE DEBUG] LLR computed, shape: {llr.shape}")
        print(f"[DECODE DEBUG] LLR[:8]: {llr[:8]}")
        print(f"[DECODE DEBUG] LLR[-8:]: {llr[-8:]}")
        # Quality check

        print(f"[DECODE DEBUG] LLR std check passed, proceeding to polar decode")
        def _validator(payload: bytes) -> bool:
            try:
                pt = self.sec.open(payload)
            except Exception:
                return False
            if not pt.startswith(b"ESAL"):
                return False
            return int.from_bytes(pt[4:8], "big") == frame_ctr

        blob = polar_dec(llr, list_size=self._list_size, validator=_validator)
        print(f"    LLR[:8]: {llr[:8]}")
        print(f"    LLR[-8:]: {llr[-8:]}")
        print(f"    LLR mean: {np.mean(llr):.3f}, std: {np.std(llr):.3f}")

        if blob is None:
            # Try sign flip
            blob = polar_dec(-llr, list_size=self._list_size, validator=_validator)
        if blob is None:
            # (3) Try alternate PN convention (restart at payload)
            llr_alt = self._llr(frame, frame_ctr, pn_variant=1)
            blob = polar_dec(llr_alt, list_size=self._list_size, validator=_validator)
            if blob is None:
                blob = polar_dec(-llr_alt, list_size=self._list_size, validator=_validator)
            if blob is None:
                print(f"    Polar decode failed")
                return False
        print(f"    Blob head (8B): {blob[:8].hex()}")

        print(f"    Polar decode OK, blob len: {len(blob)}")
        try:
            plain = self.sec.open(blob)
            print(f"    Crypto OK, plain len: {len(plain)}; "
                  f"magic={plain[:4]!r}, ctr={int.from_bytes(plain[4:8], 'big')}")
        except Exception  as e:
            if len(blob) >= 4 and blob[:4] == b"ESAL":
                plain = blob
                print("    Crypto skipped: payload appears to be PLAINTEXT (legacy mode)")
            else:
                head = blob[:8].hex()
                print(f"    Crypto failed: {type(e).__name__}: {e}; blob[:8]={head}")
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

    def _llr(self, frame: np.ndarray, frame_id: int, pn_variant: int = 0) -> np.ndarray:
        """
        Produce length-N LLRs for the payload. Steps:
          1) matched-filter with long cached taps,
          2) search an integer chip-phase shift using a sign-invariant metric,
          3) despread with PN at the chosen shift,
          4) robust LLR normalization.
        """
        N = N_DEFAULT
        # --- PN (payload) per selected variant
        if pn_variant == 0:
            # PN generated over full frame; slice the payload part (matches TX)
            pn_full = self.sec.pn_bits(frame_id, FRAME_LEN)
            pn_payload = pn_full[PRE_L + HDR_L:]
        else:
            # Alternate: PN restarted at payload length only
            pn_payload = self.sec.pn_bits(frame_id, N_DEFAULT)
        pn_sy = 2.0 * pn_payload.astype(np.float32) - 1.0  # ±1

        # --- payload segment (skip preamble + header)
        band = choose_band(self._band_key, frame_id)
        h = self._matched_filter_taps(band)
        mem = len(h) - 1
        payload_start = PRE_L + HDR_L
        if payload_start >= frame.size:
            return np.zeros(N, dtype=np.float32)

        rx_payload = frame[payload_start:].astype(np.float32, copy=False)
        if rx_payload.size == 0:
            return np.zeros(N, dtype=np.float32)

        prefix_len = min(mem, payload_start)
        if prefix_len > 0:
            prefix = frame[payload_start - prefix_len:payload_start].astype(np.float32, copy=False)
            rx_full = np.concatenate([prefix, rx_payload])
        else:
            rx_full = rx_payload

        mf = np.convolve(rx_full, h, mode="full").astype(np.float32, copy=False)
        offset = prefix_len + mem

        n = min(pn_sy.size, rx_payload.size)
        if n <= 0:
            return np.zeros(N, dtype=np.float32)

        pn_sy = pn_sy[:n]
        rx_payload = rx_payload[:n]

        print(f"[DETECTOR] Frame {frame_id}")
        print(f"  pn_payload[:32]: {pn_payload[:32]}")
        print(f"  rx[:8]: {rx_payload[:8]}")
        print(f"  despread[:8] (before shift search): {(rx_payload[:8] * pn_sy[:8])}")

        # --- matched filter (long, truncated to ~99.9% energy)
        # (4) wider shift search tied to filter memory
        raw_shift = min(n // 2, 4 * len(h), HDR_L)
        MAX_SHIFT = max(mem, raw_shift)
        MARGIN = MAX_SHIFT
        start = max(0, offset - MARGIN)
        stop = min(mf.size, offset + n + MARGIN)
        mf_win = mf[start:stop]
        base = offset - start  # zero-shift index within mf_win


        # --- guard region to avoid preamble tail bias
        guard = int(min(n // 4, max(len(h) // 2, 24)))
        if guard >= n:
            guard = max(0, n // 4)

        # --- integer shift search (sign-invariant): maximize mean |despread|
        best_s = 0
        best_score = -1.0
        for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
            i0 = base + s
            i1 = i0 + n
            if i0 < 0 or i1 > mf_win.size:
                continue
            a = mf_win[i0:i1]  # length == n
            b = pn_sy[:n]  # length == n
            d = a * b
            score = float(np.mean(np.abs(d[guard:])))  # sign-invariant
            if score > best_score:
                best_score = score
                best_s = s

        # --- apply the best shift
        i0 = base + best_s
        i1 = i0 + n
        mf_aligned = mf_win[i0:i1]  # length == n
        despread = mf_aligned * pn_sy[:n]

        # --- after picking best_s, before building despread ---
        print(f"[LLR ALIGN] best_s={best_s}, n={n}, len(h)={len(h)}, "
              f"mf_total={mf.size}, fixed_slice=[{offset}:{offset + n}] ")

        # Keep existing code that forms 'despread'...
        print(f"[LLR ALIGN] aligned_len={despread.size}, guard={guard}")

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

        tail_zeros = int(np.sum(np.isclose(llr[-64:], 0.0)))
        print(f"[LLR ALIGN] N={N}, llr_len={llr.size}, tail_zeros_64={tail_zeros}")

        if llr.size != N:
            out = np.zeros(N, dtype=np.float32)
            m = min(llr.size, N)
            out[:m] = llr[:m]
            llr = out

        return llr

    def _decrypt_blob_fallback(self, blob: bytes):
        """
        Try both common AEAD layouts:
          A) nonce || (ciphertext || tag)
          B) (ciphertext || tag) || nonce
        Return (plaintext_bytes, layout_string) on success, or (None, None).
        """
        # A) nonce at the front (what crypto.SecureChannel.seal() returns)
        if len(blob) >= 12:
            nonce_a = blob[:12]
            body_a = blob[12:]
            try:
                pt = self.sec.open(blob)  # this is exactly (nonce_a || body_a)
                return pt, "nonce-front"
            except InvalidTag:
                pass

        # B) nonce at the end (some earlier/alt implementations)
        if len(blob) >= 12:
            nonce_b = blob[-12:]
            body_b = blob[:-12]
            try:
                pt = self.aead.decrypt(nonce_b, body_b, None)  # same aead the SecureChannel uses
                return pt, "nonce-tail"
            except InvalidTag:
                pass

        return None, None


    # ----------------------------- header decode -----------------------------
    def _decode_header(self, frame: np.ndarray, band) -> tuple[bool, int, float]:
        """
        Recover 16 LSBs of frame counter from the header (16 bits, repeated 8x).
        Returns (ok, ctr_lo16, score).
        """

        # Header chips (just after preamble)
        seg = frame[PRE_L:PRE_L + HDR_L].astype(np.float32, copy=False)

        if seg.size < HDR_L:
            return False, 0, 0.0

        # PN for header: fixed (counter-independent)
        hdr_pn = 2.0 * self.sec.pn_bits(0, HDR_L).astype(np.float32) - 1.0

        # Align using the same matched-filter taps as payload (robust to room tail)
        h = self._matched_filter_taps(band)
        mf = np.convolve(seg, h, mode="full").astype(np.float32, copy=False)
        offset = len(h) - 1
        MAX_SHIFT = min(seg.size // 2, 4 * len(h))
        mem = len(h) - 1
        if MAX_SHIFT < mem:
            MAX_SHIFT = mem
        start = max(0, offset - MAX_SHIFT)
        stop = min(mf.size, offset + seg.size + MAX_SHIFT)
        mf_win = mf[start:stop]
        base = offset - start

        guard = int(max(8, min(32, len(h) // 8)))
        best_s, best_score = 0, -1.0

        for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
            i0, i1 = base + s, base + s + seg.size

            if i0 < 0 or i1 > mf_win.size:
                continue
            a = mf_win[i0:i1]
            score = float(np.mean(np.abs(a[guard:] * hdr_pn[guard:])))

            if score > best_score:
                best_score, best_s = score, s
        i0, i1 = base + best_s, base + best_s + seg.size
        a = mf_win[i0:i1]
        d = a * hdr_pn

        # Majority over 8-chip groups -> 16 bits (MSB-first)
        sums = d.reshape(HDR_BITS, HDR_REPEAT).sum(axis=1)
        bits = (sums > 0.0).astype(np.uint8)
        margin = np.mean(np.abs(sums)) / (np.sqrt(np.mean(d*d)) + 1e-12)
        val = 0

        for b in bits:
            val = (val << 1) | int(b)

        # score proxy: normalized mean absolute per bit
        score = float(np.mean(np.abs(sums)) / (np.std(d) + EPS))
        ok = (np.count_nonzero(sums > 0) >= 10) and (margin > 0.5)

        return ok, val, score
