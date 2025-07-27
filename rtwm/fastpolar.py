"""
Minimal standalone Polar Code implementation with CRC and SC-List decoding.
Inspired by Arikan's original construction. For use in real-time audio watermarking.
"""
import os
import numpy as np
from itertools import combinations

from rtwm.reliability_polar_bits import Q_Nmax


class PolarCode:
    def __init__(self, N: int, K: int, list_size: int = 1, crc_size: int = 8):
        assert (N & (N - 1)) == 0, "N must be power of 2"
        self.N = N
        self.K = K
        self.L = list_size
        self.crc_size = crc_size
        self.frozen = self._select_frozen_bits(N, K)
        self.crc_poly = 0x107  # CRC-8 poly (x^8 + x^2 + x + 1)
        print(f"[POLAR INIT] Frozen bits: {np.where(self.frozen)[0][:10]}...")
        print(f"[POLAR INIT] frozen sum: {np.sum(self.frozen)}, frozen[:10]: {np.where(self.frozen)[0][:10]}")

    def _select_frozen_bits(self, N, K):
        reliability = np.array(list(map(int, Q_Nmax.split())))  # 1024 elements
        assert len(reliability) == N, f"Q_Nmax must have {N} entries"

        frozen_mask = np.ones(N, dtype=bool)  # everything frozen by default
        frozen_mask[reliability[:K]] = False  # unfreeze the K most reliable bits
        return frozen_mask
    # def _select_frozen_bits(self, N, K):
    #     reliability = sorted(range(N), key=lambda x: bin(x).count("1"))
    #     frozen = np.ones(N, dtype=bool)
    #     for i in reliability[-K:]:
    #         frozen[i] = False
    #     return frozen

    def _crc8(self, bits):
        poly = np.uint8(self.crc_poly & 0xFF)
        reg = np.uint8(0)

        for b in bits:
            reg ^= np.uint8(b << 7)
            for _ in range(8):
                if reg & 0x80:
                    reg = np.uint8((reg << 1) ^ poly)
                else:
                    reg = np.uint8(reg << 1)

        bits8 = np.unpackbits(np.array([reg], dtype=np.uint8))
        return bits8[-8:]  # ensure exactly 8 bits

    def encode(self, bits: np.ndarray) -> np.ndarray:
        if len(bits) != self.K - self.crc_size:
            raise ValueError(f"input must be {self.K - self.crc_size} bits (K - CRC)")

        u = np.zeros(self.N, dtype=np.uint8)
        idx = np.where(~self.frozen)[0]
        crc = self._crc8(bits)
        bits_with_crc = np.concatenate((bits[:self.K - self.crc_size], crc))
        u[idx] = bits_with_crc
        return self._polar_transform(u)

    def decode(self, llr: np.ndarray):
        return self._sc_list_decode(llr)

    def _sc_list_decode(self, llr):
        # hard decision: +LLR → 1, –LLR → 0

        hard_bits = (llr > 0).astype(np.uint8)

        # F^{⊗n} is involutory over GF(2) ⇒ applying it again inverts it
        u_hat = self._polar_transform(hard_bits)
        data = u_hat[~self.frozen]  # drop frozen positions
        info, crc = data[:-self.crc_size], data[-self.crc_size:]

        ok = bool(np.all(self._crc8(info) == crc))
        if not ok:
            print("[POLAR] CRC check failed.")
        return info, ok

    def _llr_metric(self, llr, bit):
        return llr if bit else -llr

    def _polar_transform(self, u):
        N = len(u)
        stages = int(np.log2(N))
        x = u.copy()
        for s in range(stages):
            step = 2 ** (s + 1)
            for i in range(0, N, step):
                for j in range(step // 2):
                    a = x[i + j]
                    b = x[i + j + step // 2]
                    x[i + j] = a ^ b
                    x[i + j + step // 2] = b
        return x
