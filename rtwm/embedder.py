"""
Real-time frequency-hopping ultrasonic watermark transmitter.
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
    target_rel_db: float = -20.0
    N: int = N_DEFAULT
    K: int = K_DEFAULT
    preamble: np.ndarray = field(default_factory=lambda: np.array([1, 0, 1] * 21, dtype=np.uint8)[:63])

class WatermarkEmbedder:
    def __init__(self, key32: bytes, params: TxParams | None = None) -> None:
        self.p   = params or TxParams()
        self.sec = SecureChannel(key32)
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._chip_buf is None:
            self._chip_buf = np.empty(0, dtype=np.float32)
        if self._chip_buf.size < samples.size:
            # Generate new frame chips until we have at least samples.size chips
            while self._chip_buf.size < samples.size:
                frame_chips = self._make_frame_chips()
                self._chip_buf = np.concatenate((self._chip_buf, frame_chips))
            # Take the needed chips and remove them from buffer
        chips = self._chip_buf[:samples.size]
        self._chip_buf = self._chip_buf[samples.size:]
        # Mix watermark with original audio
        alpha = db_to_lin(self.p.target_rel_db)
        return samples + alpha * chips * (np.sqrt(np.mean(samples**2)) /
                                          (np.sqrt(np.mean(chips**2)) + 1e-12))

    # ------------------------------------------------------------------ internals
    def _make_frame_chips(self) -> np.ndarray:
        """Generate the watermark chips for one full frame."""
        band = choose_band(self.sec.master_key, self.frame_ctr)
        b, a = butter_bandpass(*band, self.p.fs)
        payload = self._build_payload()  # 56-byte encrypted payload
        bits = np.concatenate((self.p.preamble,
                               polar_enc(payload, N=self.p.N, K=self.p.K)))
        pn = self.sec.pn_bits(self.frame_ctr, bits.size)
        symbols = (2 * bits - 1) * (2 * pn - 1)  # BPSK modulated and spread
        chips = lfilter(b, a, symbols.astype(np.float32))
        chips /= np.max(np.abs(chips)) + 1e-12  # normalize amplitude
        self.frame_ctr += 1
        return chips.astype(np.float32)

    # def _build_payload(self) -> bytes:
    #     meta  = b"ESAL" + self.frame_ctr.to_bytes(4, "big") + secrets.token_bytes(32)
    #     blob  = self.sec.seal(meta)                     # 56 B fits K=448 bits
    #     return blob[:56]                                   # no trunc / pad

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
        rnd_len = PLAINTEXT_LEN - 8  # leave 8 bytes for nonce

        meta = (
            b"ESAL"
            + self.frame_ctr.to_bytes(4, "big")  # counter
            + secrets.token_bytes(rnd_len)  # session nonce + padding
        )

        assert len(meta) == PLAINTEXT_LEN

        blob = self.sec.seal(meta)  # 12+16+27 = 55 bytes

        assert len(blob) == 55

        return blob