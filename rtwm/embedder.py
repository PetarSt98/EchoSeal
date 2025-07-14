"""
Real-time frequency-hopping ultrasonic watermark transmitter.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, secrets
from scipy.signal import lfilter

from .utils       import choose_band, butter_bandpass, db_to_lin
from .crypto      import SecureChannel
from .polar_fast  import encode as polar_enc, N_DEFAULT, K_DEFAULT

@dataclass(slots=True)
class TxParams:
    fs: int = 48_000
    target_rel_db: float = -20.0
    N: int = N_DEFAULT
    K: int = K_DEFAULT
    preamble: np.ndarray = np.array([1, 0, 1] * 21)[:63]   # 63-chip MLS

class WatermarkEmbedder:
    def __init__(self, key32: bytes, params: TxParams | None = None) -> None:
        self.p   = params or TxParams()
        self.sec = SecureChannel(key32)
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        if self._chip_buf is None or self._chip_buf.size < samples.size:
            self._chip_buf = self._make_chips(samples.size)
        chips, self._chip_buf = self._chip_buf[:samples.size], self._chip_buf[samples.size:]

        alpha = db_to_lin(self.p.target_rel_db)
        return samples + alpha * chips * (np.sqrt(np.mean(samples**2)) /
                                          (np.sqrt(np.mean(chips**2)) + 1e-12))

    # ------------------------------------------------------------------ internals
    def _make_chips(self, out_len: int) -> np.ndarray:
        band = choose_band(self.sec.master_key, self.frame_ctr)
        b, a = butter_bandpass(*band, self.p.fs)

        payload = self._build_payload()                 # 56 bytes
        bits = np.concatenate((self.p.preamble,
                               polar_enc(payload, N=self.p.N, K=self.p.K)))

        pn   = self.sec.pn_bits(self.frame_ctr, bits.size)
        symbols = (2*bits-1) * (2*pn-1)                 # BPSK ⊗ PN

        chips = np.tile(symbols, -(-out_len // symbols.size))[:out_len].astype(np.float32)
        chips = lfilter(b, a, chips)
        chips /= np.max(np.abs(chips)) + 1e-12

        self.frame_ctr += 1
        return chips

    def _build_payload(self) -> bytes:
        meta  = b"ESAL" + self.frame_ctr.to_bytes(4, "big") + secrets.token_bytes(8)
        blob  = self.sec.seal(meta)                     # 56 B fits K=448 bits
        return blob                                     # no trunc / pad
