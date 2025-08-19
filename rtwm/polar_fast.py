"""
polar_fast – thin wrapper around rtwm.fastpolar.PolarCode
CRC-aided Chase list decoding (configurable list_size; default 8).
"""
from __future__ import annotations

import logging
from typing import Tuple, Optional, Callable

import numpy as np
from rtwm.fastpolar import PolarCode

# Defaults match your pipeline
N_DEFAULT = 1024          # codeword length
K_DEFAULT = 448           # info+CRC bits (info = 440 bits = 55 bytes)

# Cache PolarCode instances by full configuration
_cache: dict[tuple[int, int, int, int], PolarCode] = {}

def _pc(N: int, K: int, list_size: int, crc_size: int) -> PolarCode:
    key = (N, K, list_size, crc_size)
    if key not in _cache:
        _cache[key] = PolarCode(N, K, list_size=list_size, crc_size=crc_size)
    return _cache[key]

def encode(
    payload: bytes,
    *,
    N: int = N_DEFAULT,
    K: int = K_DEFAULT,
    list_size: int = 8,
    crc_size: int = 8,
    debug: bool = False
) -> np.ndarray:
    """
    Encode a 55-byte (440-bit) payload by appending CRC-8 (poly 0x07) to form K=448
    info+CRC bits, place them in the unfrozen positions, and return the length-N
    (1024) polar codeword as a 0/1 numpy array.
    """
    pc = _pc(N, K, list_size, crc_size)
    info_bytes = (pc.K - pc.crc_size) // 8
    if len(payload) != info_bytes:
        raise ValueError(f"payload must be {info_bytes} bytes (got {len(payload)})")

    bits = np.unpackbits(np.frombuffer(payload, dtype="u1"))
    if debug:
        logging.debug("[ENCODE] payload_hex=%s", payload.hex())
        logging.debug("[ENCODE] bits[:32]=%s", bits[:32])

    encoded = pc.encode(bits)  # -> length N, dtype=uint8
    if debug:
        logging.debug("[ENCODE] code[:32]=%s", encoded[:32])
    return encoded

def decode(
    llr: np.ndarray,
    *,
    N: int = N_DEFAULT,
    K: int = K_DEFAULT,
    list_size: int = 8,
    crc_size: int = 8,
    return_ok: bool = False,
    debug: bool = False,
    validator: Optional[Callable[[bytes], bool]] = None
) -> Optional[bytes] | Tuple[bytes, bool]:
    """
    Decode length-N LLRs (positive favors bit=1). On CRC pass, returns 55-byte
    payload. If `return_ok=True`, returns (payload_bytes, ok).
    """
    pc = _pc(N, K, list_size, crc_size)

    llr = np.asarray(llr)
    if llr.ndim != 1 or llr.size != pc.N:
        raise ValueError(f"LLR length {llr.size} != N {pc.N}")

    bits, ok = pc.decode(llr, validator=validator)  # -> 440 info bits (uint8), ok flag

    if debug:
        logging.debug("[DECODE] info_bits[:32]=%s ok=%s", bits[:32], ok)
        if not ok:
            # Only log length, not full data, to avoid leaks
            logging.debug("[DECODE] CRC failed; returning best candidate (len=%d bits)", bits.size)

    payload = np.packbits(bits).tobytes()  # -> 55 bytes
    if return_ok:
        return payload, ok
    return None if not ok else payload
