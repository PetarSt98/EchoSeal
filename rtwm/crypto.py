"""
ChaCha20-Poly1305 AEAD  +  PN keystream via StreamPRNG (AES-CTR).
"""
from __future__ import annotations
import secrets
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import numpy as np
from rtwm.utils import StreamPRNG, pn_bits as _pn_bits
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

class SecureChannel:
    """Encrypt / decrypt payloads and generate PN bits per frame."""
    def __init__(self, master_key: bytes) -> None:
        if len(master_key) != 32:
            raise ValueError("master_key must be 32 bytes (256 bit)")

        # Derive independent subkeys for AEAD and PRNG (domain separation)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=64,               # 32 + 32 bytes
            salt=None,               # or a fixed app-level salt if desired
            info=b"EchoSeal:KDF:v1", # domain/tag for future-proofing
        )
        okm = hkdf.derive(master_key)
        aead_key = okm[:32]
        prng_key = okm[32:]

        self._aead = ChaCha20Poly1305(aead_key)
        self._prng = StreamPRNG(prng_key)

    # ---------- watermark payload -----------------------------------------
    def seal(self, plaintext: bytes) -> bytes:
        # 12-byte (96-bit) random nonce for IETF ChaCha20-Poly1305
        nonce = secrets.token_bytes(12)
        ct = self._aead.encrypt(nonce, plaintext, b"")
        return nonce + ct

    def open(self, blob: bytes) -> bytes:  # raises on failure
        if len(blob) < 12 + 16:
            raise ValueError("ciphertext too short")
        nonce, ct = blob[:12], blob[12:]
        return self._aead.decrypt(nonce, ct, b"")

    # ---------- PRNG interface --------------------------------------------
    def pn_bits(self, frame_ctr: int, n_bits: int) -> np.ndarray:
        """Return `n_bits` pseudo-random bits for this frame."""
        return _pn_bits(self._prng, frame_ctr, n_bits)
