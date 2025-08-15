"""
Fixed watermark embedder with consistent filtering approach.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, secrets
from scipy.signal import lfilter
import logging
from rtwm.utils       import choose_band, butter_bandpass, db_to_lin
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import encode as polar_enc, N_DEFAULT, K_DEFAULT

EPS = 1e-12
MIN_RMS_SILENCE = 1e-4   # ~ -80 dBFS; gate watermark in near-silence
MIX_HEADROOM = 0.98

@dataclass(slots=True)
class TxParams:
    fs: int = 48_000
    target_rel_db: float = -10.0
    N: int = N_DEFAULT
    K: int = K_DEFAULT
    preamble: np.ndarray = field(default_factory=lambda: np.array([1, 0, 1] * 21, dtype=np.uint8)[:63])

class WatermarkEmbedder:
    def __init__(self, key32: bytes, params: TxParams | None = None) -> None:
        self.p   = params or TxParams()
        self.sec = SecureChannel(key32)
        self._band_key = getattr(self.sec, "band_key", key32)
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None
        self._session_nonce = secrets.token_bytes(8)
        self._bp_state: dict[tuple[int, int], np.ndarray] = {}

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Mix one block of watermark chips into `samples` at a level tied to host RMS,
        with silence gating and a clip-safe headroom limiter."""
        if self._chip_buf is None:
            self._chip_buf = np.empty(0, dtype=np.float32)

        in_rms = float(np.sqrt(np.mean(samples.astype(np.float32, copy=False) ** 2)))
        if in_rms < MIN_RMS_SILENCE:
            return samples

        needed = samples.size
        while self._chip_buf.size < needed:
            # logging.debug("[TX] generating frame %d", self.frame_ctr)
            frame_chips = self._make_frame_chips()
            self._chip_buf = np.concatenate((self._chip_buf, frame_chips))
            self.frame_ctr = (self.frame_ctr + 1) % (2 ** 32)

        chips = self._chip_buf[:needed].astype(np.float32, copy=False)
        self._chip_buf = self._chip_buf[needed:]

        # Base scale proportional to host RMS and clip-safe headroom
        alpha = db_to_lin(self.p.target_rel_db)
        scale = alpha * in_rms

        headroom = MIX_HEADROOM - float(np.max(np.abs(samples)))
        if headroom < 0.0:
            headroom = 0.0
        peak = float(np.max(np.abs(chips))) + EPS
        scale = min(scale, headroom / peak) if peak > 0.0 else 0.0

        return samples + chips * scale

    # ------------------------------------------------------------------ internals
    def _make_frame_chips(self) -> np.ndarray:
        """Generate the watermark chips for one full frame with consistent filtering."""
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
        pre_bits = self.mseq_63()  # length 63
        preamble_symbols = 2.0 * pre_bits.astype(np.float32) - 1.0

        data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0

        frame_len = pre_bits.size + data_bits.size  # 63 + 1024 = 1087
        pn_full = self.sec.pn_bits(self.frame_ctr, frame_len)  # 1087 bits
        pn_payload = pn_full[pre_bits.size:]  # last 1024 bits
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        # Spread payload and concatenate with unspread preamble
        spread_payload = data_symbols * pn_symbols
        symbols = np.concatenate((preamble_symbols, spread_payload)).astype(np.float32, copy=False)

        # --- FILTER WITH PERSISTENT STATE ---
        b, a = butter_bandpass(*band, self.p.fs, order=4)
        zi_len = max(len(a), len(b)) - 1
        zi = self._bp_state.get(band)
        # ensure zi exists and has matching dtype/length
        zi_dtype = np.result_type(a, b, symbols)
        if zi is None or zi.shape[0] != zi_len or zi.dtype != zi_dtype:
            zi = np.zeros(zi_len, dtype=zi_dtype)

        chips, zf = lfilter(b, a, symbols, zi=zi)
        self._bp_state[band] = zf

        # --- STEADY-STATE NORMALIZATION (avoid transient bias) ---
        start = max(16, zi_len)  # tie to filter state length
        steady = chips[start:] if chips.size > start else chips
        energy = float(np.mean(steady ** 2))
        if energy > EPS:
            chips /= np.sqrt(energy)

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

    def mseq_63(self) -> np.ndarray:
        # x^6 + x + 1 primitive polynomial (one of the standard taps for 63)
        reg = np.array([1, 1, 1, 1, 1, 1], dtype=np.uint8)
        out = np.empty(63, dtype=np.uint8)
        for i in range(63):
            out[i] = reg[-1]
            fb = reg[-1] ^ reg[0]  # taps at [6,1]
            reg[1:] = reg[:-1]
            reg[0] = fb
        return out
