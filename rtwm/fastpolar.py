from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional, Sequence, Tuple
import numpy as np
from rtwm.reliability_polar_bits import Q_Nmax

def _parse_reliability_indices(N: int) -> np.ndarray:
    rel = np.fromiter((int(x) for x in Q_Nmax.split()), dtype=np.int64)
    if rel.size != N:
        raise ValueError(f"Q_Nmax must have {N} entries (has {rel.size})")
    if np.any(rel < 0) or np.any(rel >= N) or np.unique(rel).size != N:
        raise ValueError("Q_Nmax must be a permutation of 0..N-1")
    return rel

@dataclass(slots=True)
class PolarCode:
    N: int
    K: int                   # info + CRC bits
    list_size: int = 8
    crc_size: int = 8
    debug: bool = False

    # ---- predeclare slot-backed internals ----
    _crc_poly: np.uint8 = field(init=False, repr=False, default=np.uint8(0x07))
    frozen: np.ndarray = field(init=False, repr=False, default=None)
    _data_pos: np.ndarray = field(init=False, repr=False, default=None)
    _u_buf: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError("N must be a power of 2 and > 0")
        if not (0 < self.K <= self.N):
            raise ValueError("0 < K <= N must hold")
        if self.list_size < 1:
            raise ValueError("list_size must be >= 1")
        if not (0 < self.crc_size < self.K):
            raise ValueError("0 < crc_size < K must hold")

        # _crc_poly already declared with default; keep as 0x07
        rel = _parse_reliability_indices(self.N)

        # frozen mask (True=frozen), unfreeze K most reliable
        self.frozen = np.ones(self.N, dtype=bool)
        self.frozen[rel[-self.K:]] = False

        self._data_pos = np.flatnonzero(~self.frozen)
        if self._data_pos.size != self.K:
            raise RuntimeError("Internal error: data positions != K")

        self._u_buf = np.empty(self.N, dtype=np.uint8)

    # --------------- API ---------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        if info_bits.dtype != np.uint8:
            info_bits = info_bits.astype(np.uint8, copy=False)
        if info_bits.ndim != 1:
            raise ValueError("info_bits must be a 1D array")
        if info_bits.size != self.K - self.crc_size:
            raise ValueError(f"info_bits must have length {self.K - self.crc_size}")

        crc = self._crc8(info_bits)
        # NOTE: concatenate then cast (dtype kwarg isn’t portable)
        data = np.concatenate((info_bits, crc)).astype(np.uint8, copy=False)

        u = self._u_buf
        u.fill(0)
        u[self._data_pos] = data
        return self._polar_transform(u)

    def decode(self, llr: np.ndarray) -> Tuple[np.ndarray, bool]:
        if llr.ndim != 1 or llr.size != self.N:
            raise ValueError(f"llr must be 1D length {self.N}")

        llr = llr.astype(np.float64, copy=False)

        # hard decision → invert → CRC
        hard = (llr > 0.0).astype(np.uint8)
        u_hat = self._polar_transform(hard)
        u_hat[self.frozen] = 0
        data_hat = u_hat[self._data_pos]
        info0 = data_hat[: self.K - self.crc_size]
        crc0  = data_hat[self.K - self.crc_size : self.K]
        if self._crc_ok(info0, crc0):
            return info0.copy(), True

        if self.list_size == 1:
            return info0.copy(), False

        # Chase: flip least-reliable codeword positions
        order = np.argsort(np.abs(llr))                 # least reliable first
        cand_positions = order[:min(16, self.N)]        # cap surface
        emitted = 1
        best_ok = None
        best_any = (np.inf, info0.copy())

        for r in range(1, min(5, cand_positions.size) + 1):  # up to 5-bit flips
            for combo in combinations(cand_positions, r):
                idx = np.fromiter(combo, dtype=np.int64)
                metric = float(np.abs(llr[idx]).sum())
                if best_ok is not None and metric >= best_ok[0]:
                    continue

                x2 = hard.copy()
                x2[idx] ^= 1
                u2 = self._polar_transform(x2)
                u2[self.frozen] = 0  # <-- NEW: enforce code constraint
                d2 = u2[self._data_pos]
                info2 = d2[: self.K - self.crc_size]
                crc2  = d2[self.K - self.crc_size : self.K]

                if self._crc_ok(info2, crc2):
                    if best_ok is None or metric < best_ok[0]:
                        best_ok = (metric, info2.copy())
                else:
                    if metric < best_any[0]:
                        best_any = (metric, info2.copy())

                emitted += 1
                if emitted >= self.list_size:
                    break
            if emitted >= self.list_size:
                break

        if best_ok is not None:
            return best_ok[1], True
        return best_any[1], False

    # --------------- internals ---------------
    def _crc8(self, bits: np.ndarray) -> np.ndarray:
        reg = np.uint8(0)
        b = bits.astype(np.uint8, copy=False)
        for bit in b:
            reg ^= np.uint8((bit & 1) << 7)
            if reg & 0x80:
                reg = np.uint8(((reg << 1) ^ self._crc_poly) & 0xFF)
            else:
                reg = np.uint8((reg << 1) & 0xFF)
        return np.unpackbits(np.array([reg], dtype=np.uint8))

    def _crc_ok(self, info: np.ndarray, crc_bits: np.ndarray) -> bool:
        return bool(np.all(self._crc8(info) == crc_bits))

    @staticmethod
    def _polar_transform(u: np.ndarray) -> np.ndarray:
        N = u.size
        x = u.copy()
        stages = int(np.log2(N))
        for s in range(stages):
            step = 1 << (s + 1)
            half = step >> 1
            for i in range(0, N, step):
                a = x[i : i + half]
                b = x[i + half : i + step]
                x[i : i + half] = a ^ b
                x[i + half : i + step] = b
        return x
