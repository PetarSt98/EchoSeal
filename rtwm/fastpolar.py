"""
CRC-aided polar code (CA-SCL) used for the EchoSeal watermark payload.

Encoder / decoder for a length-N polar code (N a power of two, N ≤ 1024) whose
information set is taken from the 5G NR reliability sequence (3GPP TS 38.212,
Table 5.3.1.2-1).  ``K`` counts information *plus* CRC-8 bits.  Decoding is
successive-cancellation list (SCL) with CRC selection; the caller may supply an
additional ``validator`` (EchoSeal passes the AEAD open/verify) that acts as
the final arbiter among CRC-passing candidates.

LLR convention throughout:  llr = log P(bit=1) / P(bit=0)   (positive ⇒ 1).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from rtwm.reliability_polar_bits import Q_Nmax

N_MAX = 1024

# Parse and sanity-check the reliability table once at import time.  The 3GPP
# sequence lists channel indices from least to most reliable.
_Q1024 = np.fromiter((int(x) for x in Q_Nmax.split()), dtype=np.int64)
if _Q1024.size != N_MAX or np.unique(_Q1024).size != N_MAX:
    raise ImportError("Q_Nmax must be a permutation of 0..1023")


def _reliability(N: int) -> np.ndarray:
    """Reliability-ordered channel indices for block length N (least→most).

    Standard 5G NR nesting: keep the entries of the master sequence that are
    smaller than N, preserving their order.
    """
    return _Q1024[_Q1024 < N]


# ─────────────────────────── SC message passing ───────────────────────────
def _f_function(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact check-node (boxplus) combine for log(P1/P0) LLRs."""
    return np.logaddexp(a, b) - np.logaddexp(0.0, a + b)


def _g_function(a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Variable-node combine given the decided left-child bits ``u``."""
    return b + (1.0 - 2.0 * u.astype(np.float64, copy=False)) * a


def _metric_penalty(llr: float, bit: int) -> float:
    """Negative log-likelihood added to a path metric for assigning ``bit``."""
    abs_llr = abs(llr)
    penalty = float(np.log1p(np.exp(-abs_llr)))
    preferred = 1 if llr >= 0.0 else 0
    if bit != preferred:
        penalty += abs_llr
    return penalty


# ───────────────────────────── SCL path state ─────────────────────────────
class _SharedArray:
    """Reference-counted array wrapper enabling copy-on-write between paths."""

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
    """One decoding path: lazily computed LLRs (alpha) and partial sums (beta).

    The per-level alpha/beta arrays are shared between cloned paths and copied
    only on write, which keeps list decoding memory- and copy-efficient.
    """

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

    # ---- copy-on-write helpers -------------------------------------------
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

    def clone(self) -> "_ListPath":
        child = _ListPath.__new__(_ListPath)
        child.n = self.n
        child.N = self.N
        child.metric = self.metric
        child.u = self.u.copy()
        child.alpha = list(self.alpha)
        for shared in child.alpha:
            shared.acquire()
        child.alpha_valid = [arr.copy() for arr in self.alpha_valid]
        child.beta = list(self.beta)
        for shared in child.beta:
            shared.acquire()
        return child

    # ---- message passing ---------------------------------------------------
    def calc_llr(self, bit_index: int) -> float:
        return float(self._calc_alpha(self.n, bit_index)[0])

    def _calc_alpha(self, level: int, node: int) -> np.ndarray:
        start, end = self._slice(level, node)
        if self.alpha_valid[level][node]:
            return self._alpha_ro(level)[start:end]

        if level == 0:
            self.alpha_valid[0][0] = True
            return self._alpha_ro(0)[start:end]

        parent = self._calc_alpha(level - 1, node // 2)
        half = parent.size // 2
        left, right = parent[:half], parent[half:]
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
        """Fix bit ``bit_index`` to ``bit_value`` and update partial sums."""
        b = np.uint8(bit_value & 1)
        self.u[bit_index] = b

        level = self.n
        node = bit_index
        start, end = self._slice(level, node)
        self._beta_rw(level)[start:end] = b
        self.alpha_valid[level][node] = False

        # Fold completed right children into their parents.
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
            self.alpha_valid[level][node] = False

        # Alpha caches along the path to the root are now stale.
        while level > 0:
            node //= 2
            level -= 1
            self.alpha_valid[level][node] = False


# ─────────────────────────────── PolarCode ────────────────────────────────
class PolarCode:
    """CRC-8-aided polar code with SCL decoding."""

    CRC_BITS = 8
    _CRC_POLY = 0x07  # CRC-8: x^8 + x^2 + x + 1, init 0, MSB-first

    def __init__(self, N: int, K: int, *, list_size: int = 8, crc_size: int = 8) -> None:
        if N <= 0 or N & (N - 1):
            raise ValueError("N must be a power of two")
        if N > N_MAX:
            raise ValueError(f"N must be <= {N_MAX} (reliability table limit)")
        if crc_size != self.CRC_BITS:
            raise ValueError("only CRC-8 is supported")
        if not crc_size < K <= N:
            raise ValueError("crc_size < K <= N must hold")
        if list_size < 1:
            raise ValueError("list_size must be >= 1")

        self.N = N
        self.K = K
        self.list_size = list_size
        self.crc_size = crc_size

        # Information set = the K *most* reliable synthetic channels.  The
        # 3GPP sequence is ordered least → most reliable, hence the last K.
        rel = _reliability(N)
        self.frozen = np.ones(N, dtype=bool)
        self.frozen[rel[-K:]] = False
        self._data_pos = np.flatnonzero(~self.frozen)

        self._n = int(np.log2(N))
        self._info_len = K - crc_size

    # ------------------------------------------------------------------ API
    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """Encode ``K - 8`` information bits into a length-N 0/1 codeword."""
        info_bits = np.asarray(info_bits)
        if info_bits.ndim != 1 or info_bits.size != self._info_len:
            raise ValueError(f"info_bits must be 1D with length {self._info_len}")
        info_bits = info_bits.astype(np.uint8, copy=False)

        u = np.zeros(self.N, dtype=np.uint8)
        u[self._data_pos] = np.concatenate((info_bits, self._crc8(info_bits)))
        return self._polar_transform(u)

    def decode(
        self,
        llr: np.ndarray,
        validator: Callable[[bytes], bool] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """SCL-decode length-N LLRs; returns ``(info_bits, ok)``.

        ``ok`` is True only if a candidate passed the CRC (and ``validator``
        when given).  With ``ok`` False the best-effort bits are returned.
        """
        llr = np.asarray(llr, dtype=np.float64)
        if llr.ndim != 1 or llr.size != self.N:
            raise ValueError(f"llr must be 1D with length {self.N}")

        def accepted(info: np.ndarray) -> bool:
            if validator is None:
                return True
            try:
                return bool(validator(np.packbits(info).tobytes()))
            except Exception:
                return False

        # Fast path: hard decision + inverse transform (the transform is an
        # involution).  Succeeds on clean channels and costs only O(N log N).
        u_hat = self._polar_transform((llr > 0.0).astype(np.uint8))
        data0 = u_hat[self._data_pos]
        info0 = data0[: self._info_len]
        crc0 = data0[self._info_len :]
        if (
            not u_hat[self.frozen].any()
            and self._crc_ok(info0, crc0)
            and accepted(info0)
        ):
            return info0.copy(), True

        # Full SCL decoding.
        paths: list[_ListPath] = [_ListPath(llr, self._n)]

        for bit_index in range(self.N):
            if self.frozen[bit_index]:
                for path in paths:
                    path.metric += _metric_penalty(path.calc_llr(bit_index), 0)
                    path.extend(bit_index, 0)
                continue

            candidates: list[tuple[float, int, int]] = []
            for idx, path in enumerate(paths):
                llr_val = path.calc_llr(bit_index)
                candidates.append((path.metric + _metric_penalty(llr_val, 0), idx, 0))
                candidates.append((path.metric + _metric_penalty(llr_val, 1), idx, 1))

            candidates.sort(key=lambda item: item[0])
            survivors = candidates[: self.list_size]

            clone_budget: dict[int, int] = defaultdict(int)
            for _, idx, _ in survivors:
                clone_budget[idx] += 1

            # Pre-create clones so the original object serves one survivor.
            cloned: dict[int, list[_ListPath]] = {
                idx: [paths[idx].clone() for _ in range(count - 1)]
                for idx, count in clone_budget.items()
                if count > 1
            }

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
                if old_idx not in clone_budget:
                    path.release()

            paths = new_paths

        # Pick the best candidate: CRC+validator pass wins outright; otherwise
        # remember the best CRC-passing and best overall paths as fallbacks.
        best_crc: tuple[float, np.ndarray] | None = None
        best_any: tuple[float, np.ndarray] = (np.inf, info0.copy())

        for path in sorted(paths, key=lambda p: p.metric):
            data = path.u[self._data_pos]
            info_bits = data[: self._info_len].copy()
            crc_bits = data[self._info_len :]

            if self._crc_ok(info_bits, crc_bits):
                if accepted(info_bits):
                    return info_bits, True
                if best_crc is None or path.metric < best_crc[0]:
                    best_crc = (path.metric, info_bits)
            elif path.metric < best_any[0]:
                best_any = (path.metric, info_bits)

        if best_crc is not None:
            return best_crc[1], False
        return best_any[1], False

    # ------------------------------------------------------------ internals
    def _crc8(self, bits: np.ndarray) -> np.ndarray:
        reg = 0
        for bit in bits:
            reg ^= (int(bit) & 1) << 7
            reg = ((reg << 1) ^ self._CRC_POLY) & 0xFF if reg & 0x80 else (reg << 1) & 0xFF
        return np.unpackbits(np.array([reg], dtype=np.uint8))

    def _crc_ok(self, info: np.ndarray, crc_bits: np.ndarray) -> bool:
        return bool(np.all(self._crc8(info) == crc_bits))

    @staticmethod
    def _polar_transform(u: np.ndarray) -> np.ndarray:
        """Apply G_N = F^{⊗n} (its own inverse over GF(2)), vectorised."""
        x = u.copy()
        half = 1
        while half < x.size:
            blk = x.reshape(-1, 2 * half)
            blk[:, :half] ^= blk[:, half:]
            half *= 2
        return x
