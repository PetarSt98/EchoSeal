"""
XChaCha20-Poly1305 AEAD  +  AES-CTR PN keystream.
"""
from __future__ import annotations
import secrets
from cryptography.hazmat.primitives.ciphers.aead import XChaCha20Poly1305
import numpy as np
from .utils import StreamPRNG, pn_bits as _pn_bits

class SecureChannel:
    """Encrypt / decrypt payloads and generate PN bits per frame."""
    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("master_key must be 32 bytes (256 bit)")
        self.master_key = master_key
        self._aead = XChaCha20Poly1305(master_key)
        self._prng = StreamPRNG(master_key)

    # ---------- watermark payload -----------------------------------------
    def seal(self, plaintext: bytes) -> bytes:
        nonce = secrets.token_bytes(24)                 # 192-bit nonce
        return nonce + self._aead.encrypt(nonce, plaintext, b"")

    def open(self, blob: bytes) -> bytes:               # raises on failure
        return self._aead.decrypt(blob[:24], blob[24:], b"")

    # ---------- PRNG interface --------------------------------------------
    def pn_bits(self, frame_ctr: int, n_bits: int) -> np.ndarray:
        """Return `n_bits` pseudo-random bits for this frame."""
        return _pn_bits(self._prng, frame_ctr, n_bits)
