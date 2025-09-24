from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from rtwm.reliability_polar_bits import Q_Nmax

def _parse_reliability_indices(N: int) -> np.ndarray:
    rel = np.fromiter((int(x) for x in Q_Nmax.split()), dtype=np.int64)
    if rel.size != N:
        raise ValueError(f"Q_Nmax must have {N} entries (has {rel.size})")
    if np.any(rel < 0) or np.any(rel >= N) or np.unique(rel).size != N:
        raise ValueError("Q_Nmax must be a permutation of 0..N-1")
    return rel

def _f_function(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact f-combine for LLR vectors (element-wise)."""

    # Our LLRs are defined as log(P(bit=1)/P(bit=0)), so the synthetic channel
    # update for the "left" child is logaddexp(a, b) - logaddexp(0, a + b).
    return np.logaddexp(a, b) - np.logaddexp(0.0, a + b)


def _g_function(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Exact g-combine for LLR vectors given left child partial sums."""

    return b + (1.0 - 2.0 * u.astype(np.float64, copy=False)) * a


def _metric_penalty(llr_scalar: float, bit: int) -> float:
    """Return the negative log-likelihood contribution for assigning ``bit``."""

    abs_llr = abs(llr_scalar)
    penalty = float(np.log1p(np.exp(-abs_llr)))
    preferred = 1 if llr_scalar >= 0.0 else 0
    if bit != preferred:
        penalty += abs_llr
    return penalty


class _SharedArray:
    __slots__ = ("arr", "refcount")

    def __init__(self, arr: np.ndarray) -> None:
        self.arr = arr
        self.refcount = 1

    def acquire(self) -> None:
        self.refcount += 1

    def release(self) -> None:
        self.refcount -= 1
        if self.refcount < 0:
            raise RuntimeError("Shared array released too many times")


class _ListPath:
    """State container for a single path inside the SCL decoder."""

    __slots__ = ("n", "N", "metric", "u", "alpha", "alpha_valid", "beta")

    def __init__(self, llr: np.ndarray, n: int) -> None:
        self.n = n
        self.N = llr.size
        self.metric = 0.0
        self.u = np.zeros(self.N, dtype=np.uint8)

        self.alpha = [_SharedArray(np.zeros(self.N, dtype=np.float64)) for _ in range(n + 1)]
        self.alpha[0].arr[:] = llr
        self.alpha_valid = [np.zeros(1 << level, dtype=bool) for level in range(n + 1)]
        self.alpha_valid[0][0] = True
        self.beta = [_SharedArray(np.zeros(self.N, dtype=np.uint8)) for _ in range(n + 1)]

    # ---- helpers ---------------------------------------------------------
    def _slice(self, level: int, node: int) -> tuple[int, int]:
        step = 1 << (self.n - level)
        start = node * step
        return start, start + step

    def _alpha_ro(self, level: int) -> np.ndarray:
        return self.alpha[level].arr

    def _alpha_rw(self, level: int) -> np.ndarray:
        shared = self.alpha[level]
        if shared.refcount > 1:
            shared.release()
            shared = _SharedArray(shared.arr.copy())
            self.alpha[level] = shared
        return shared.arr

    def _beta_ro(self, level: int) -> np.ndarray:
        return self.beta[level].arr

    def _beta_rw(self, level: int) -> np.ndarray:
        shared = self.beta[level]
        if shared.refcount > 1:
            shared.release()
            shared = _SharedArray(shared.arr.copy())
            self.beta[level] = shared
        return shared.arr

    def release(self) -> None:
        for shared in self.alpha:
            shared.release()
        for shared in self.beta:
            shared.release()

    # ---- path management -------------------------------------------------
    def clone(self) -> "_ListPath":
        child = _ListPath.__new__(_ListPath)
        child.n = self.n
        child.N = self.N
        child.metric = self.metric
        child.u = self.u.copy()
        child.alpha = [shared for shared in self.alpha]
        for shared in child.alpha:
            shared.acquire()
        child.alpha_valid = [arr.copy() for arr in self.alpha_valid]
        child.beta = [shared for shared in self.beta]
        for shared in child.beta:
            shared.acquire()
        return child

    # ---- message passing -------------------------------------------------
    def calc_llr(self, bit_index: int) -> float:
        segment = self._calc_alpha(self.n, bit_index)
        return float(segment[0])

    def _calc_alpha(self, level: int, node: int) -> np.ndarray:
        start, end = self._slice(level, node)
        if self.alpha_valid[level][node]:
            return self._alpha_ro(level)[start:end]

        if level == 0:
            self.alpha_valid[0][0] = True
            return self._alpha_ro(0)[start:end]

        parent_segment = self._calc_alpha(level - 1, node // 2)
        half = parent_segment.size // 2
        left = parent_segment[:half]
        right = parent_segment[half:]
        dest = self._alpha_rw(level)[start:end]

        if node % 2 == 0:
            dest[:] = _f_function(left, right)
        else:
            left_start, left_end = self._slice(level, node - 1)
            beta_left = self._beta_ro(level)[left_start:left_end]
            dest[:] = _g_function(left, right, beta_left)

        self.alpha_valid[level][node] = True
        return dest

    def extend(self, bit_index: int, bit_value: int) -> None:
        b = np.uint8(bit_value & 1)
        self.u[bit_index] = b

        level = self.n
        node = bit_index
        start, end = self._slice(level, node)
        self._beta_rw(level)[start:end] = b
        self.alpha_valid[level][node] = False

        while node % 2 == 1 and level > 0:
            left_node = node - 1
            parent_node = node // 2
            level -= 1
            parent_start, parent_end = self._slice(level, parent_node)
            left_start, left_end = self._slice(level + 1, left_node)
            right_start, right_end = self._slice(level + 1, node)

            half = (parent_end - parent_start) // 2
            left_bits = self._beta_ro(level + 1)[left_start:left_end]
            right_bits = self._beta_ro(level + 1)[right_start:right_end]
            parent_beta = self._beta_rw(level)
            parent_beta[parent_start : parent_start + half] = left_bits ^ right_bits
            parent_beta[parent_start + half : parent_end] = right_bits

            node = parent_node
            start, end = parent_start, parent_end
            self.alpha_valid[level][node] = False

        # Invalidate ancestors on the path
        temp_level, temp_node = level, node
        while temp_level > 0:
            temp_node //= 2
            temp_level -= 1
            self.alpha_valid[temp_level][temp_node] = False


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
    _n: int = field(init=False, repr=False, default=0)
    _info_len: int = field(init=False, repr=False, default=0)

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

        # frozen mask (True=frozen). ``Q_Nmax`` follows the 5G NR convention
        # where indices are sorted from most → least reliable, so the first
        # ``K`` entries form the information set.
        self.frozen = np.ones(self.N, dtype=bool)
        self.frozen[rel[: self.K]] = False

        self._data_pos = np.flatnonzero(~self.frozen)
        if self._data_pos.size != self.K:
            raise RuntimeError("Internal error: data positions != K")

        self._u_buf = np.empty(self.N, dtype=np.uint8)
        self._n = int(np.log2(self.N))
        self._info_len = self.K - self.crc_size

    # --------------- API ---------------
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        if info_bits.dtype != np.uint8:
            info_bits = info_bits.astype(np.uint8, copy=False)
        if info_bits.ndim != 1:
            raise ValueError("info_bits must be a 1D array")
        if info_bits.size != self._info_len:
            raise ValueError(f"info_bits must have length {self._info_len}")

        crc = self._crc8(info_bits)
        # NOTE: concatenate then cast (dtype kwarg isn’t portable)
        data = np.concatenate((info_bits, crc)).astype(np.uint8, copy=False)

        u = self._u_buf
        u.fill(0)
        u[self._data_pos] = data
        return self._polar_transform(u)

    def decode(self, llr: np.ndarray, validator: Optional[Callable[[bytes], bool]] = None) -> Tuple[np.ndarray, bool]:
        if llr.ndim != 1 or llr.size != self.N:
            raise ValueError(f"llr must be 1D length {self.N}")

        llr = llr.astype(np.float64, copy=False)

        # hard decision → invert → CRC
        hard = (llr > 0.0).astype(np.uint8)
        u_hat = self._polar_transform(hard)
        u_hat[self.frozen] = 0
        data_hat = u_hat[self._data_pos]
        info0 = data_hat[: self._info_len]
        crc0 = data_hat[self._info_len : self.K]

        if self._crc_ok(info0, crc0):
            if validator is not None:
                try:
                    if validator(np.packbits(info0).tobytes()):
                        return info0.copy(), True
                except Exception:
                    pass
            else:
                return info0.copy(), True

        paths: list[_ListPath] = [_ListPath(llr, self._n)]

        for bit_index in range(self.N):
            if self.frozen[bit_index]:
                for path in paths:
                    llr_val = path.calc_llr(bit_index)
                    path.metric += _metric_penalty(llr_val, 0)
                    path.extend(bit_index, 0)
                continue

            candidates: list[tuple[float, int, int]] = []
            for idx, path in enumerate(paths):
                llr_val = path.calc_llr(bit_index)
                base_metric = path.metric
                candidates.append((base_metric + _metric_penalty(llr_val, 0), idx, 0))
                candidates.append((base_metric + _metric_penalty(llr_val, 1), idx, 1))

            if not candidates:
                return info0.copy(), False

            candidates.sort(key=lambda item: item[0])
            survivors = candidates[: self.list_size]

            clone_budget: dict[int, int] = defaultdict(int)
            for _, idx, _ in survivors:
                clone_budget[idx] += 1

            survivor_indices = set(clone_budget)

            cloned: dict[int, list[_ListPath]] = {}
            for idx, count in clone_budget.items():
                if count > 1:
                    base_path = paths[idx]
                    # Pre-create clones so we preserve the original for one survivor
                    cloned[idx] = [base_path.clone() for _ in range(count - 1)]

            used_primary: dict[int, bool] = {idx: False for idx in clone_budget}
            new_paths: list[_ListPath] = []
            for metric, idx, bit_value in survivors:
                if not used_primary[idx]:
                    path = paths[idx]
                    used_primary[idx] = True
                else:
                    path = cloned[idx].pop()
                path.metric = metric
                path.extend(bit_index, bit_value)
                new_paths.append(path)

            for old_idx, path in enumerate(paths):
                if old_idx not in survivor_indices:
                    path.release()

            paths = new_paths

        best_crc: Optional[Tuple[float, np.ndarray]] = None
        best_any = (np.inf, info0.copy())

        for path in sorted(paths, key=lambda p: p.metric):
            data = path.u[self._data_pos]
            info_bits = data[: self._info_len].copy()
            crc_bits = data[self._info_len : self.K]

            metric = path.metric
            if self._crc_ok(info_bits, crc_bits):
                if validator is not None:
                    try:
                        if validator(np.packbits(info_bits).tobytes()):
                            return info_bits, True
                    except Exception:
                        pass
                else:
                    return info_bits, True

                if best_crc is None or metric < best_crc[0]:
                    best_crc = (metric, info_bits)
            elif metric < best_any[0]:
                best_any = (metric, info_bits)

        if best_crc is not None:
            return best_crc[1], False

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
