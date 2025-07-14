"""
AES-GCM convenience wrapper (single responsibility: crypto only).
"""

from __future__ import annotations
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets, os

class AESCipher:
    """Encrypt / decrypt small messages with a static key."""

    def __init__(self, key: bytes) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("AES key must be 128/192/256-bit")
        self._aes = AESGCM(key)

    # ---------- public API ----------

    def encrypt(self, plaintext: bytes) -> bytes:
        """Return `iv || ciphertext || tag`."""
        iv = secrets.token_bytes(12)           # 96-bit IV
        return iv + self._aes.encrypt(iv, plaintext, None)

    def decrypt(self, blob: bytes) -> bytes:
        """Opposite of encrypt(); raises on failure."""
        iv, ct_tag = blob[:12], blob[12:]
        return self._aes.decrypt(iv, ct_tag, None)
