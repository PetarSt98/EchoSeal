"""
Authenticated encryption (ChaCha20-Poly1305) plus a keyed PN source.

A single 256-bit master key is expanded with HKDF-SHA256 into three independent
sub-keys so the AEAD, the PN keystream and the frequency-hopping schedule never
share key material.
"""
from __future__ import annotations

import secrets

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from rtwm.utils import StreamPRNG, pn_bits as _pn_bits

_NONCE_LEN = 12  # IETF ChaCha20-Poly1305 nonce
_TAG_LEN = 16    # Poly1305 tag
_KDF_INFO = b"EchoSeal:KDF:v1"


class SecureChannel:
    """Seal/open watermark payloads and derive per-frame PN bits."""

    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("master_key must be 32 bytes (256 bit)")

        # Domain-separated sub-keys: AEAD | PN | band-hop schedule.
        okm = HKDF(
            algorithm=hashes.SHA256(),
            length=96,
            salt=None,
            info=_KDF_INFO,
        ).derive(master_key)

        self._aead = ChaCha20Poly1305(okm[:32])
        self._prng = StreamPRNG(okm[32:64])
        self.band_key = okm[64:96]  # public: non-secret keyed hop selection

    # ---------------------------------------------------------------- AEAD
    def seal(self, plaintext: bytes) -> bytes:
        """Encrypt and authenticate: returns ``nonce || ciphertext || tag``."""
        nonce = secrets.token_bytes(_NONCE_LEN)
        return nonce + self._aead.encrypt(nonce, plaintext, b"")

    def open(self, blob: bytes) -> bytes:
        """Inverse of :meth:`seal`; raises ``InvalidTag`` on tamper."""
        if len(blob) < _NONCE_LEN + _TAG_LEN:
            raise ValueError("ciphertext too short")
        nonce, ct = blob[:_NONCE_LEN], blob[_NONCE_LEN:]
        return self._aead.decrypt(nonce, ct, b"")

    # ----------------------------------------------------------------- PN
    def pn_bits(self, frame_ctr: int, n_bits: int) -> np.ndarray:
        """Return `n_bits` deterministic PN bits (uint8 {0,1}) for this frame."""
        return _pn_bits(self._prng, frame_ctr, n_bits)
