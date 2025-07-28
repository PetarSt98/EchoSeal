"""
Fixed watermark detector with proper preamble template generation.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter, correlate

from rtwm.utils       import BAND_PLAN, butter_bandpass, resample_to, choose_band
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import decode as polar_dec, N_DEFAULT

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
        """Test a raw frame directly (for debugging)."""
        # For raw frames that are already filtered and normalized,
        # we should test them directly without additional filtering
        print(f"[RAW FRAME TEST] Testing frame of length {len(signal)}")

        if len(signal) == FRAME_LEN:
            # This is a complete frame - try to decode it directly
            for ctr in range(4):  # Try a few frame counters
                llr = self._llr(signal, ctr)

                # Quality check
                llr_std = np.std(llr)
                print(f"[RAW TEST] ctr={ctr}: LLR std={llr_std:.3f}")

                if llr_std < 0.3:
                    continue

                blob = polar_dec(llr)
                if blob is None:
                    continue

                try:
                    plain = self.sec.open(blob)
                    if plain.startswith(b"ESAL"):
                        embedded_ctr = int.from_bytes(plain[4:8], "big")
                        if embedded_ctr == ctr:
                            print(f"[SUCCESS] Raw frame decoded with ctr={ctr}")
                            return True
                except:
                    continue

        # If direct testing fails, we might have an unfiltered frame
        # Try the band scanning approach
        band = choose_band(self.sec.master_key, 0)
        return self._scan_band(signal, band, skip_filtering=False)

    def _scan_band(self, signal: np.ndarray, band, skip_filtering=False) -> bool:
        """Scan for watermark frames in the given frequency band."""
        print(f"[SCAN-BAND] Band {band}, signal_len={len(signal)}")

        # Apply bandpass filter to signal
        b, a = butter_bandpass(*band, self.fs_target)
        y = lfilter(b, a, signal.astype(np.float32))

        # CRITICAL: Create preamble template the SAME way as TX
        # The TX applies the filter to the ENTIRE frame (preamble + payload)
        # So we need to create a template that matches what the filtered preamble looks like

        # First, create a dummy frame to extract just the preamble part
        preamble_symbols = 2.0 * PREAMBLE.astype(np.float32) - 1.0

        # Create a full dummy frame (preamble + zeros for payload)
        dummy_frame = np.zeros(FRAME_LEN, dtype=np.float32)
        dummy_frame[:len(PREAMBLE)] = preamble_symbols

        # Filter the entire dummy frame (same as TX does)
        dummy_filtered = lfilter(b, a, dummy_frame)

        # Extract just the preamble part after filtering
        preamble_template = dummy_filtered[:len(PREAMBLE)].copy()

        # Normalize the template (TX normalizes the whole frame, but we only care about preamble correlation)
        template_energy = np.sum(preamble_template**2)
        if template_energy > 1e-12:
            preamble_template /= np.sqrt(template_energy)

        print(f"[SCAN-BAND] Preamble template: mean={np.mean(preamble_template):.6f}, std={np.std(preamble_template):.3f}")

        # Now search for this template in the signal
        # Use 'valid' mode to avoid edge effects
        if len(y) < len(preamble_template):
            print(f"[SCAN-BAND] Signal too short for correlation")
            return False

        corr = correlate(y, preamble_template, mode='valid')

        # Normalize correlation by local signal energy
        corr_normalized = np.zeros_like(corr)
        for i in range(len(corr)):
            segment = y[i:i+len(preamble_template)]
            seg_energy = np.sum(segment**2)
            temp_energy = np.sum(preamble_template**2)

            if seg_energy > 1e-12 and temp_energy > 1e-12:
                corr_normalized[i] = corr[i] / np.sqrt(seg_energy * temp_energy)

        # Find peaks
        threshold = 0.5  # Correlation threshold
        peaks = np.where(np.abs(corr_normalized) > threshold)[0]

        print(f"[SCAN-BAND] Found {len(peaks)} peaks above threshold {threshold}")

        if len(peaks) == 0:
            # Lower threshold and try again
            threshold = 0.3
            peaks = np.where(np.abs(corr_normalized) > threshold)[0]
            print(f"[SCAN-BAND] With lower threshold {threshold}: {len(peaks)} peaks")

        # Try each peak
        for peak_idx in peaks[:20]:  # Try up to 20 peaks
            frame_start = peak_idx  # In 'valid' mode, this is the actual start position

            if frame_start + FRAME_LEN > len(y):
                continue

            frame = y[frame_start:frame_start+FRAME_LEN]

            # Estimate frame counter
            est_ctr = frame_start // FRAME_LEN

            # Try decoding
            if self._try_window(frame, est_ctr, TIGHT_DELTA):
                print(f"[SUCCESS] Frame found at position {frame_start}")
                return True

        return False

    # ------------------------------------------------------------------ window search
    def _try_window(self, frame: np.ndarray, ctr0: int, delta: int) -> bool:
        """Try different frame counter values within window."""
        for ctr in range(max(0, ctr0 - delta), ctr0 + delta + 1):
            llr = self._llr(frame, ctr)

            # Quality check
            llr_std = np.std(llr)
            if llr_std < 0.3:
                print(f"[RX] ctr={ctr}: LLR variance too low ({llr_std:.3f}), skipping")
                continue

            print(f"[LLR] ctr={ctr}: range=[{llr.min():.3f}, {llr.max():.3f}], std={llr_std:.3f}")

            blob = polar_dec(llr)
            if blob is None:
                print(f"[RX] ctr={ctr}: polar decode failed")
                continue

            try:
                plain = self.sec.open(blob)
                print(f"[RX] ctr={ctr}: decrypt success, checking payload")
            except Exception as e:
                print(f"[RX] ctr={ctr}: decrypt failed ({type(e).__name__})")
                continue

            if not plain.startswith(b"ESAL"):
                print(f"[RX] ctr={ctr}: bad prefix: {plain[:4]}")
                continue

            embedded_ctr = int.from_bytes(plain[4:8], "big")
            if embedded_ctr != ctr:
                print(f"[RX] ctr={ctr}: counter mismatch (embedded={embedded_ctr})")
                continue

            nonce = plain[8:16]
            if self.session_nonce and nonce == self.session_nonce:
                print(f"[SUCCESS] ctr={ctr}: watermark verified (repeat nonce)")
                return True
            elif self.session_nonce is None:
                self.session_nonce = nonce
                print(f"[SUCCESS] ctr={ctr}: watermark verified (new nonce)")
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
        pn_payload = pn_full[len(PREAMBLE):]

        # Extract payload part
        payload_received = frame[len(PREAMBLE):].copy()

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

        print(f"[LLR] ctr={frame_id}: noise_est={noise_estimate:.3f}")
        print(f"[LLR] final: range=[{llr.min():.3f}, {llr.max():.3f}], mean={np.mean(llr):.3f}, std={np.std(llr):.3f}")

        return llr