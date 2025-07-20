"""
fastpolar wrapper – CRC-aided SC-List (L = 8).
"""
from __future__ import annotations
import numpy as np
from rtwm.fastpolar import PolarCode

N_DEFAULT = 1024         # block
K_DEFAULT = 448          # coded bits; information = 440 bits (55 bytes)

_cache: dict[tuple[int, int], PolarCode] = {}

def _pc(N: int, K: int) -> PolarCode:
    key = (N, K)
    if key not in _cache:
        _cache[key] = PolarCode(N, K, list_size=8, crc_size=8)
    return _cache[key]

# def encode(payload: bytes, *, N=N_DEFAULT, K=K_DEFAULT) -> np.ndarray:
#     if len(payload) * 8 != K:
#         raise ValueError("payload size mismatch")
#     return _pc(N, K).encode(np.unpackbits(np.frombuffer(payload, "u1")))

def encode(payload: bytes, *, N=N_DEFAULT, K=K_DEFAULT) -> np.ndarray:
    """
    Encode a 55-byte payload (440 information bits) and append an 8-bit CRC.
    The resulting codeword has length ``K`` = 448 bits (1024 after polar
    ‘channel’ coding).
    """
    pc           = _pc(N, K)
    info_bytes   = (pc.K - pc.crc_size) // 8        # 440 bits → 55 bytes
    if len(payload) != info_bytes:
        raise ValueError(f"payload must be {info_bytes} bytes (got {len(payload)})")

    bits = np.unpackbits(np.frombuffer(payload, dtype="u1"))
    return pc.encode(bits)

def decode(llr: np.ndarray, *, N=N_DEFAULT, K=K_DEFAULT) -> bytes | None:
    dec, ok = _pc(N, K).decode(llr)
    return None if not ok else np.packbits(dec).tobytes()
