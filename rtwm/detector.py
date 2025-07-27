"""
Offline detector – sync, soft LLR, SCL-list polar, adaptive threshold.
FIXED VERSION with proper signal processing that matches TX exactly.
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
        print("Detector hop0:", choose_band(key32, 0))

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
    def verify_raw_frame(self, signal: np.ndarray) -> bool:
        """Bypass filtering when testing synthetic chip frames."""
        band = choose_band(self.sec.master_key, 0)
        return self._scan_band(signal, band, skip_filtering=True)

    def _scan_band(self, signal: np.ndarray, band, skip_filtering=False) -> bool:
        """Enhanced band scanning with better debugging."""
        if skip_filtering:
            y = signal.astype(np.float32)
            tpl = (2 * PREAMBLE - 1).astype(np.float32)
        else:
            b, a = butter_bandpass(*band, self.fs_target)
            y = lfilter(b, a, signal.astype(np.float32))
            tpl = lfilter(b, a, (2 * PREAMBLE - 1).astype(np.float32))

        tpl /= np.sqrt(np.mean(tpl ** 2)) + 1e-12  # normalize energy
        corr = correlate(y, tpl, mode="valid")

        PEAK_SHIFT = (len(PREAMBLE) - 1) // 2
        thresh = 3 * np.std(corr)
        MAX_PEAKS = 200
        peaks = np.where(corr > thresh)[0][:MAX_PEAKS]

        print(f"Band {band}: {len(peaks)} peaks detected, max corr: {np.max(corr) if len(corr) > 0 else 0}")
        print(f"[DEBUG] Correlation stats: mean={np.mean(corr):.3f}, std={np.std(corr):.3f}, thresh={thresh:.3f}")

        # Also try the strongest peaks regardless of threshold
        if len(peaks) == 0:
            # Find top 10 peaks even if below threshold
            peak_indices = np.argsort(corr)[-10:]
            peaks = peak_indices[corr[peak_indices] > 0]
            print(f"[DEBUG] No peaks above threshold, trying top {len(peaks)} peaks")

        wide_done = False
        for i, p in enumerate(peaks):
            if p + FRAME_LEN > y.size:
                continue

            print(f"[DEBUG] Testing peak {i+1}/{len(peaks)} at position {p}, corr={corr[p]:.3f}")

            start = p - PEAK_SHIFT
            if start < 0:
                print(f"[DEBUG] Peak too close to start: p={p}, PEAK_SHIFT={PEAK_SHIFT}")
                continue

            if start + FRAME_LEN > y.size:
                print(f"[DEBUG] Peak too close to end: start={start}, FRAME_LEN={FRAME_LEN}, y.size={y.size}")
                continue

            frame = y[start: start + FRAME_LEN]
            est_ctr = start // FRAME_LEN

            print(f"[DEBUG] Frame extracted: len={len(frame)}, est_ctr={est_ctr}")

            # 1) fast ±3 window
            if self._try_window(frame, est_ctr, TIGHT_DELTA):
                return True

            # 2) one-time wider fallback
            if not wide_done and self._try_window(frame, est_ctr, WIDE_DELTA):
                return True
            wide_done = True

            # Stop after testing first few strong peaks to avoid excessive computation
            if i >= 5:
                print("[DEBUG] Tested first 5 peaks, moving on")
                break

        return False

    # ------------------------------------------------------------------ window search
    def _try_window(self, frame: np.ndarray, ctr0: int, delta: int) -> bool:
        for ctr in range(max(0, ctr0 - delta), ctr0 + delta + 1):
            llr = self._llr(frame, ctr)
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
        """
        CORRECTLY FIXED LLR calculation for multiplication-based spreading with proper scaling.

        TX does:
        1. Map data bits to symbols: data_symbols = (2*data_bits - 1)
        2. Map PN bits to symbols: pn_symbols = (2*pn_bits - 1)
        3. Multiply: spread_symbols = data_symbols * pn_symbols
        4. Apply bandpass filter (reduces amplitude!)

        RX must:
        1. Get received symbols (after filtering, amplitude reduced)
        2. Despread: despread_symbols = received_symbols * pn_symbols
        3. Scale and calculate LLR with proper amplitude compensation
        """

        # Get PN sequence for this frame
        pn_full = self.sec.pn_bits(frame_id, FRAME_LEN)
        pn_payload = pn_full[len(PREAMBLE):]

        # Extract payload from received frame
        payload_received = frame[len(PREAMBLE):].copy()

        print(f"[DEBUG] Frame {frame_id}: processing {len(payload_received)} payload samples")

        # Ensure we don't exceed array bounds
        min_len = min(len(payload_received), len(pn_payload))
        payload_received = payload_received[:min_len]
        pn_payload = pn_payload[:min_len]

        # Convert PN bits to symbols: {0,1} → {-1,+1}
        pn_symbols = 2 * pn_payload.astype(np.float32) - 1

        # Despread by multiplying with PN symbols
        despread_symbols = payload_received * pn_symbols

        # CRITICAL: Proper amplitude scaling
        # The bandpass filter significantly reduces signal amplitude
        # We need to estimate and compensate for this

        # Method 1: Adaptive scaling based on despread signal statistics
        despread_power = np.mean(np.abs(despread_symbols))
        expected_power = 1.0  # We expect symbols close to ±1

        if despread_power > 1e-6:  # Avoid division by zero
            amplitude_scale = expected_power / despread_power
        else:
            amplitude_scale = 1.0

        # Apply amplitude compensation
        scaled_despread = despread_symbols * amplitude_scale

        # Method 2: Also try scaling based on signal energy
        signal_rms = np.sqrt(np.mean(payload_received ** 2))
        if signal_rms > 1e-6:
            energy_scale = 1.0 / signal_rms
            energy_scaled_despread = despread_symbols * energy_scale
        else:
            energy_scaled_despread = despread_symbols

        # Use the scaling method that gives more reasonable values
        if np.std(scaled_despread) > np.std(energy_scaled_despread):
            final_despread = scaled_despread
            scale_method = "amplitude"
        else:
            final_despread = energy_scaled_despread
            scale_method = "energy"

        # Convert to LLR with aggressive scaling
        # LLR should be large enough for polar decoder to work
        base_scale = 8.0  # Increased from 4.0

        # Additional adaptive scaling based on signal variance
        signal_std = np.std(final_despread)
        if signal_std > 1e-6:
            adaptive_scale = 2.0 / signal_std  # Target std ≈ 2.0
            adaptive_scale = np.clip(adaptive_scale, 1.0, 50.0)  # Reasonable bounds
        else:
            adaptive_scale = 10.0

        total_scale = base_scale * adaptive_scale
        llr = final_despread * total_scale

        # Ensure we have the right length for polar decoder
        if len(llr) < len(pn_payload):
            # Pad with zeros (neutral LLR)
            llr_full = np.zeros(len(pn_payload), dtype=np.float32)
            llr_full[:len(llr)] = llr
            llr = llr_full

        # Clip to prevent overflow in polar decoder
        llr = np.clip(llr, -30.0, 30.0)

        print(f"[LLR SCALING] ctr={frame_id}, method={scale_method}, "
              f"amp_scale={amplitude_scale:.3f}, adaptive_scale={adaptive_scale:.3f}, total_scale={total_scale:.3f}")
        print(f"[LLR FINAL] ctr={frame_id}, "
              f"llr_range=[{llr.min():.3f}, {llr.max():.3f}], "
              f"llr_mean={np.mean(llr):.3f}, llr_std={np.std(llr):.3f}")

        # Check if LLR values look reasonable for polar decoding
        if np.std(llr) < 1.0:
            print("[WARNING] LLR variance still low - polar decoder may struggle")
        elif np.std(llr) > 3.0:
            print("[INFO] LLR variance looks good for polar decoding")
        else:
            print("[INFO] LLR variance moderate - should work")

        return llr