"""
Fixed watermark embedder with consistent filtering approach.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, secrets
from scipy.signal import lfilter
import logging
from rtwm.utils import choose_band, butter_bandpass, db_to_lin, mseq_63
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import encode as polar_enc, N_DEFAULT, K_DEFAULT

EPS = 1e-12
MIN_RMS_SILENCE = 1e-4   # ~ -80 dBFS; gate watermark in near-silence
MIX_HEADROOM = 0.98
HDR_BITS    = 16
HDR_REPEAT  = 8
HDR_L       = 128

@dataclass(slots=True)
class TxParams:
    fs: int = 48_000
    target_rel_db: float = -10.0
    floor_rel_dbfs: float = -35.0
    N: int = N_DEFAULT
    K: int = K_DEFAULT
    preamble: np.ndarray = field(default_factory=lambda: mseq_63())

class WatermarkEmbedder:
    def __init__(self, key32: bytes, params: TxParams | None = None) -> None:
        self.p   = params or TxParams()
        self.sec = SecureChannel(key32)
        self._band_key = getattr(self.sec, "band_key", key32)
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None
        self._session_nonce = secrets.token_bytes(8)
        # Pre-compute the static sequences we reuse every frame so there is no
        # chance of drifting from the detector's expectations.
        self._preamble_sy = 2.0 * self.p.preamble.astype(np.float32) - 1.0
        # Header PN never depends on the frame counter; cache the ±1 symbols.
        self._hdr_pn_sy = 2.0 * self.sec.pn_bits(0, HDR_L).astype(np.float32) - 1.0

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Mix watermark chips into `samples` at a level tied to host RMS with a
        clip-safe headroom limiter and an absolute floor so silence still carries WM."""
        if self._chip_buf is None:
            self._chip_buf = np.empty(0, dtype=np.float32)

        x = samples.astype(np.float32, copy=False)
        in_rms = float(np.sqrt(np.mean(x * x)) + EPS)

        needed = samples.size
        while self._chip_buf.size < needed:
            # logging.debug("[TX] generating frame %d", self.frame_ctr)
            frame_chips = self._make_frame_chips()
            self._chip_buf = np.concatenate((self._chip_buf, frame_chips))
            self.frame_ctr = (self.frame_ctr + 1) % (2 ** 32)

        chips = self._chip_buf[:needed].astype(np.float32, copy=False)
        self._chip_buf = self._chip_buf[needed:]

        # Base scale proportional to host RMS, with an absolute floor for silence
        alpha = db_to_lin(self.p.target_rel_db)
        scale_host = alpha * in_rms
        scale_floor = db_to_lin(self.p.floor_rel_dbfs)  # absolute vs. FS
        scale = max(scale_host, scale_floor)

        headroom = MIX_HEADROOM - float(np.max(np.abs(x)))
        if headroom < 0.0:
            headroom = 0.0
        peak = float(np.max(np.abs(chips))) + EPS
        scale = min(scale, headroom / peak) if peak > 0.0 else 0.0

        return x + chips * scale

    # ------------------------------------------------------------------ internals
    def _make_frame_chips(self) -> np.ndarray:
        """Generate watermark chips for one full frame (preamble + header + payload)."""
        # Pick a stable band key (prefer derived; fall back to legacy attr if present)
        try:
            band = choose_band(self._band_key, self.frame_ctr)
        except Exception as e:
            import logging
            logging.exception("[TX] Error choosing band for frame %d", self.frame_ctr)
            raise

        # Build payload (55 bytes) and (optional) debug print of first few bits
        payload = self._build_payload()
        # optional debug: show first 16 bits only
        import logging
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype="u1"))
        logging.debug("[TX] bits[:16]=%s len=%d", payload_bits[:16], payload_bits.size)

        # Polar encode to N chips (bits) and map to BPSK symbols
        data_bits = polar_enc(payload, N=self.p.N, K=self.p.K)  # -> 1024 bits

        # --- PREAMBLE & PN SIZING (use MLS length consistently) ---
        pre_bits = self.p.preamble.astype(np.uint8) # length 63
        preamble_symbols = self._preamble_sy

        data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0

        # ---- (1A) COUNTER-BOOTSTRAP HEADER (16 bits, each repeated 8x) ----
        ctr_lo16 = np.uint16(self.frame_ctr & 0xFFFF)
        ctr_bytes = np.array([ctr_lo16 >> 8, ctr_lo16 & 0xFF], dtype=np.uint8)
        hdr_bits = np.unpackbits(ctr_bytes)  # 16 bits, MSB-first
        hdr_bits_rep = np.repeat(hdr_bits, HDR_REPEAT)  # 16*8 = 128 bits
        hdr_bpsk = 2.0 * hdr_bits_rep.astype(np.float32) - 1.0
        hdr_sy = hdr_bpsk * self._hdr_pn_sy

        frame_len = pre_bits.size + HDR_L + data_bits.size  # 63 + 128 + 1024 = 1215
        pn_full = self.sec.pn_bits(self.frame_ctr, frame_len)  # 1215 bits
        pn_payload = pn_full[pre_bits.size + HDR_L:]  # last 1024 bits
        if pn_payload.size != data_bits.size:
            raise RuntimeError(
                f"PN payload length {pn_payload.size} != encoded payload {data_bits.size}"
            )
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        # Spread payload and concatenate with unspread preamble
        spread_payload = data_symbols * pn_symbols
        symbols = np.concatenate((preamble_symbols, hdr_sy, spread_payload)).astype(np.float32, copy=False)
        if symbols.size != frame_len:
            raise RuntimeError(
                f"Frame assembled to {symbols.size} chips, expected {frame_len}"
            )

        if self.frame_ctr < 3:  # Only log first few frames
            print(f"[EMBEDDER] Frame {self.frame_ctr}")
            print(f"  pn_payload[:32]: {pn_payload[:32]}")
            print(f"  data_symbols[:8]: {data_symbols[:8]}")
            print(f"  spread_payload[:8]: {spread_payload[:8]}")
            print(f"  symbols[:8]: {symbols[:8]}")  # This is after concatenation

        # --- (2) FILTER ALIGNMENT: zero-state at preamble; continue header+payload with that zi ---
        b, a = butter_bandpass(*band, self.p.fs, order=4)
        zi0_len = max(len(a), len(b)) - 1
        zi0 = np.zeros(zi0_len, dtype=np.result_type(a, b, symbols))
        # filter preamble from zero state
        y_pre, zi1 = lfilter(b, a, preamble_symbols, zi=zi0)
        # then header + payload continue from preamble's end state
        y_rest, _ = lfilter(b, a, np.concatenate((hdr_sy, spread_payload)).astype(np.float32, copy=False), zi=zi1)
        chips = np.concatenate((y_pre, y_rest))

        # --- STEADY-STATE NORMALIZATION (avoid transient bias) ---
        peak_val = float(np.max(np.abs(chips))) + EPS
        if peak_val > 3.0:  # Only scale if unreasonably large
            chips = chips * (1.0 / peak_val)

        return chips.astype(np.float32, copy=False)

    def _build_payload(self) -> bytes:
        """
        Build the 27-byte plaintext, seal it with XChaCha20-Poly1305 and
        return the resulting 55-byte ciphertext ‖ tag ‖ nonce.
        """
        meta = (
                b"ESAL"  # 4 bytes
                + self.frame_ctr.to_bytes(4, "big")  # 4 bytes
                + self._session_nonce  # 8 bytes (fixed per session)
                + secrets.token_bytes(11)  # 11 bytes random padding
        )

        assert len(meta) == 27
        blob = self.sec.seal(meta)
        assert len(blob) == 55
        return blob
