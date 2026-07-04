"""
Byte-level convenience wrapper around :class:`rtwm.fastpolar.PolarCode`.

The embedder calls :func:`encode` with the 55-byte sealed payload; the
detector calls :func:`decode` with length-N LLRs (positive favours bit = 1).
Code instances are cached per configuration.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rtwm.fastpolar import PolarCode

N_DEFAULT = 1024                       # codeword length (chips per payload)
K_DEFAULT = 448                        # information + CRC bits
CRC_BITS = PolarCode.CRC_BITS          # 8
PAYLOAD_BYTES = (K_DEFAULT - CRC_BITS) // 8  # 55

_cache: dict[tuple[int, int, int], PolarCode] = {}


def _pc(N: int, K: int, list_size: int) -> PolarCode:
    key = (N, K, list_size)
    if key not in _cache:
        _cache[key] = PolarCode(N, K, list_size=list_size)
    return _cache[key]


def encode(payload: bytes, *, N: int = N_DEFAULT, K: int = K_DEFAULT) -> np.ndarray:
    """Encode `payload` into a length-N 0/1 codeword (CRC-8 appended)."""
    pc = _pc(N, K, 1)
    info_bytes = (pc.K - pc.crc_size) // 8
    if len(payload) != info_bytes:
        raise ValueError(f"payload must be {info_bytes} bytes (got {len(payload)})")
    return pc.encode(np.unpackbits(np.frombuffer(payload, dtype="u1")))


def decode(
    llr: np.ndarray,
    *,
    N: int = N_DEFAULT,
    K: int = K_DEFAULT,
    list_size: int = 8,
    return_ok: bool = False,
    validator: Callable[[bytes], bool] | None = None,
) -> bytes | None | tuple[bytes, bool]:
    """Decode length-N LLRs back into the payload bytes.

    Returns the payload on success and ``None`` on failure, or
    ``(payload, ok)`` when ``return_ok`` is set.  ``validator`` (payload ->
    bool) selects among CRC-passing SCL candidates, e.g. AEAD verification.
    """
    pc = _pc(N, K, list_size)
    bits, ok = pc.decode(llr, validator=validator)
    payload = np.packbits(bits).tobytes()
    if return_ok:
        return payload, ok
    return payload if ok else None
