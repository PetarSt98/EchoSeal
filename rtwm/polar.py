"""
Tiny polar (N, K) encoder suitable for ≤512 block lengths.
Only encoding is needed on TX side.
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- helpers
def _bitrev(i: int, bits: int) -> int:
    return int(f"{i:0{bits}b}"[::-1], 2)

def _frozen_mask(N: int, K: int) -> np.ndarray:
    """
    Return 1-D mask where 1 = frozen, 0 = info bits.
    Here we use a crude Bhattacharyya-order approximation suitable
    for speech watermarking (N ≤ 512).  Replace with DE or GA tables
    for production.
    """
    reliability = sorted(range(N), key=lambda x: bin(x).count("1"))
    info_idx    = reliability[-K:]
    mask        = np.ones(N, dtype=np.uint8)
    mask[info_idx] = 0
    return mask

def _polar_transform(u: np.ndarray) -> np.ndarray:
    N = u.size
    if N == 1:
        return u
    even = _polar_transform((u[::2] ^ u[1::2]) & 1)
    odd  = _polar_transform(u[1::2])
    return np.concatenate((even, odd))

# --------------------------------------------------------------------------- public
def polar_encode(payload: bytes, *, N: int = 512, K: int = 344) -> np.ndarray:
    """
    Map `K` info bits into an `N`-length polar codeword (BPSK ready: 0/1).
    `len(payload) * 8` **must equal** `K`.
    """
    m_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if m_bits.size != K:
        raise ValueError(f"payload={m_bits.size} bits, expected {K}")
    u       = np.zeros(N, dtype=np.uint8)
    frozen  = _frozen_mask(N, K)
    u[frozen == 0] = m_bits
    x = _polar_transform(u) & 1
    return x.astype(np.uint8)

def polar_decode(code: np.ndarray, *, N: int = 512, K: int = 344) -> bytes:
    """
    *Hard-decision* decode: apply inverse transform, strip frozen bits.
    Works well because upstream encryption + PN spreading already gives
    very low raw BER; for adversarial/noisy channels swap in SC decoder.
    """
    if code.size != N:
        raise ValueError("wrong codeword length")
    u = _polar_transform(code & 1) & 1
    info = u[_frozen_mask(N, K) == 0]
    return np.packbits(info).tobytes()
