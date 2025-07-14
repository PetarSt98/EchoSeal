"""
Spread-spectrum ultrasonic watermark transmitter (TX).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from .utils  import butter_bandpass, pseudorandom_chips, db_to_lin, keyed_seed
from .crypto import AESCipher
from .polar  import polar_encode

# --------------------------------------------------------------------------- config
@dataclass(slots=True)
class TxParams:
    fs: int            = 48_000
    band: tuple[int,int] = (18_000, 22_000)
    target_rel_db: float = -20.0      # watermark level vs. speech RMS
    N: int             = 512          # polar block
    K: int             = 344          # info bits (=43 bytes)

# --------------------------------------------------------------------------- embedder
class WatermarkEmbedder:
    """
    `process(chunk)` -> returns same length float32 with hidden watermark.
    Stateless w.r.t. audio I/O; maintains its own frame counter internally.
    """

    def __init__(self, key: bytes, params: TxParams | None = None) -> None:
        self.p     = params or TxParams()
        self.aes   = AESCipher(key)
        self.b_bp, self.a_bp = butter_bandpass(*self.p.band, self.p.fs)
        self.frame_ctr = 0
        self._chip_buf: np.ndarray | None = None   # carry-over when chunk≠codeword

    # ---------- public ------------------------------------------------------
    def process(self, speech: np.ndarray) -> np.ndarray:
        """
        Mix watermark into `speech` (mono float32) and return same-size array.
        """
        if self._chip_buf is None or self._chip_buf.size < speech.size:
            self._chip_buf = self._make_chips(speech.size)

        chips, self._chip_buf = (
            self._chip_buf[: speech.size],
            self._chip_buf[speech.size :],
        )

        # scale watermark to keep power `target_rel_db` below speech RMS
        sp_rms = np.sqrt(np.mean(speech**2) + 1e-12)
        wm_rms = np.sqrt(np.mean(chips**2))
        alpha  = db_to_lin(self.p.target_rel_db) * sp_rms / (wm_rms + 1e-12)

        return speech + alpha * chips

    # ---------- internal ----------------------------------------------------
    def _make_chips(self, min_len: int) -> np.ndarray:
        """
        Produce ≥min_len BPSK chips band-limited to ultrasonic band.
        """
        payload = self._build_payload()
        bits    = polar_encode(payload, N=self.p.N, K=self.p.K)
        # Repeat bits until we exceed min_len (wrap-around for streaming)
        reps    = -(-min_len // bits.size)  # ceil division
        bits    = np.tile(bits, reps)[: min_len]

        seed    = keyed_seed(self.aes.encrypt(b"seed")[:16], self.frame_ctr)
        pn      = pseudorandom_chips(seed, bits.size)
        chips   = pn * (2 * bits - 1).astype(np.int8)  # map {0,1}→{-1,+1}

        # ultrasonic band-pass
        chips_f = lfilter(self.b_bp, self.a_bp, chips.astype(np.float32))
        chips_f /= np.max(np.abs(chips_f)) + 1e-12
        self.frame_ctr += 1
        return chips_f

    def _build_payload(self) -> bytes:
        """
        Assemble <session-id ‖ frame-counter> and encrypt.
        Payload length = K/8 bytes.
        """
        # 4-byte session (constant for run) + 4-byte frame counter
        meta = (b"RTWM" + self.frame_ctr.to_bytes(4, "big"))
        blob = self.aes.encrypt(meta)
        # ensure exactly K bits
        need = self.p.K // 8
        if len(blob) >= need:
            return blob[:need]          # truncate *ciphertext end* (keeps IV+tag)
        return blob.ljust(need, b"\0")  # pad unlikely edge
