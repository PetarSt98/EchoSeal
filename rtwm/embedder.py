"""
Real-time frequency-hopping ultrasonic watermark transmitter.
FIXED VERSION - corrected symbol mapping and spreading operation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, secrets
from scipy.signal import lfilter

from rtwm.utils       import choose_band, butter_bandpass, db_to_lin
from rtwm.crypto      import SecureChannel
from rtwm.polar_fast  import encode as polar_enc, N_DEFAULT, K_DEFAULT

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
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None
        print("Embedder band frame 0:", choose_band(self.sec.master_key, 0))

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._chip_buf is None:
            self._chip_buf = np.empty(0, dtype=np.float32)

        needed = samples.size
        while self._chip_buf.size < needed:
            frame_chips = self._make_frame_chips()  # ~1087 chips
            self._chip_buf = np.concatenate((self._chip_buf, frame_chips))

        chips = self._chip_buf[:needed]
        self._chip_buf = self._chip_buf[needed:]
        print(f"[TX] frame {self.frame_ctr} — chips: {len(chips)}")
        alpha = db_to_lin(self.p.target_rel_db)
        in_rms = np.sqrt(np.mean(samples ** 2))
        if in_rms < 1e-12:
            scale = 1.0
        else:
            scale = in_rms / (np.sqrt(np.mean(chips ** 2)) + 1e-12)

        return samples + alpha * chips * scale

    # ------------------------------------------------------------------ internals
    def _make_frame_chips(self) -> np.ndarray:
        """Generate the watermark chips for one full frame."""
        band = choose_band(self.sec.master_key, self.frame_ctr)
        b, a = butter_bandpass(*band, self.p.fs)

        payload = self._build_payload()  # 55-byte encrypted payload
        payload_bits = np.unpackbits(np.frombuffer(payload, dtype="u1"))
        print(f"[TX] bits: {payload_bits[:16]}... len={len(payload_bits)}")

        data_bits = polar_enc(payload, N=self.p.N, K=self.p.K)  # 1024
        frame_len = len(self.p.preamble) + data_bits.size  # 1087

        pn_full = self.sec.pn_bits(self.frame_ctr, frame_len)  # 1087 bits
        pn_payload = pn_full[len(self.p.preamble):]  # last 1024 bits

        # FIXED: Consistent symbol mapping and spreading
        # Map bits to BPSK symbols: {0,1} → {-1,+1}
        preamble_symbols = 2.0 * self.p.preamble.astype(np.float32) - 1.0
        data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        # Spread data by multiplying with PN sequence (element-wise)
        spread_payload = data_symbols * pn_symbols

        # Concatenate preamble (unspread) + spread payload
        symbols = np.concatenate((preamble_symbols, spread_payload))

        # Apply bandpass filter
        chips = lfilter(b, a, symbols)

        # Normalize to unit energy
        energy = np.mean(chips ** 2)
        if energy > 1e-12:
            chips /= np.sqrt(energy)

        print(f"[TX] frame {self.frame_ctr}, chip len = {len(chips)}")
        self.frame_ctr += 1
        return chips.astype(np.float32)

    def _build_payload(self) -> bytes:
        """
        Build the 27-byte plaintext, seal it with XChaCha20-Poly1305 and
        return the resulting 55-byte ciphertext ‖ tag ‖ nonce.

        Layout on the wire (55 bytes):
            0‥11   : 12-byte XChaCha nonce
           12‥27   : 16-byte Poly1305 tag
           28‥54   : 27-byte ciphertext
        """
        PLAINTEXT_LEN = 27  # 55-(12+16)
        rnd_len = PLAINTEXT_LEN - 8  # leave 8 bytes for counter + padding

        meta = (
            b"ESAL"
            + self.frame_ctr.to_bytes(4, "big")  # counter
            + secrets.token_bytes(rnd_len)  # session nonce + padding
        )

        assert len(meta) == PLAINTEXT_LEN
        blob = self.sec.seal(meta)  # 12+16+27 = 55 bytes
        assert len(blob) == 55
        return blob