"""
fastpolar wrapper – CRC-aided SC-List (L = 8).
"""
from __future__ import annotations
import numpy as np
from fastpolar import PolarCode

N_DEFAULT = 1024         # block
K_DEFAULT = 448          # info bits (56 bytes – fits sealed blob)

_cache: dict[tuple[int, int], PolarCode] = {}

def _pc(N: int, K: int) -> PolarCode:
    key = (N, K)
    if key not in _cache:
        _cache[key] = PolarCode(N, K, list_size=8, crc_size=8)
    return _cache[key]

def encode(payload: bytes, *, N=N_DEFAULT, K=K_DEFAULT) -> np.ndarray:
    if len(payload) * 8 != K:
        raise ValueError("payload size mismatch")
    return _pc(N, K).encode(np.unpackbits(np.frombuffer(payload, "u1")))

def decode(llr: np.ndarray, *, N=N_DEFAULT, K=K_DEFAULT) -> bytes | None:
    dec, ok = _pc(N, K).decode(llr)
    return None if not ok else np.packbits(dec).tobytes()
