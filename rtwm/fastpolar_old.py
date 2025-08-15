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
        frozen_mask[reliability[-K:]] = False  # unfreeze the K most reliable bits
        return frozen_mask
    # def _select_frozen_bits(self, N, K):
    #     reliability = sorted(range(N), key=lambda x: bin(x).count("1"))
    #     frozen = np.ones(N, dtype=bool)
    #     for i in reliability[-K:]:
    #         frozen[i] = False
    #     return frozen

    def _crc8(self, bits):
        poly = np.uint8(self.crc_poly & 0xFF)  # effective 0x07
        reg = np.uint8(0)
        b = np.asarray(bits, dtype=np.uint8)
        for bit in b:
            # XOR incoming bit into MSB
            reg ^= np.uint8((bit & 1) << 7)
            # single bit-time update
            if reg & 0x80:
                reg = np.uint8(((reg << 1) ^ poly) & 0xFF)
            else:
                reg = np.uint8((reg << 1) & 0xFF)
        return np.unpackbits(np.array([reg], dtype=np.uint8))

    def encode(self, bits: np.ndarray) -> np.ndarray:
        print(f"[PolarCode] Input bits: {bits[:32]}")
        if len(bits) != self.K - self.crc_size:
            raise ValueError(f"input must be {self.K - self.crc_size} bits (K - CRC)")

        u = np.zeros(self.N, dtype=np.uint8)
        idx = np.where(~self.frozen)[0]
        crc = self._crc8(bits)
        print(f"[PolarCode] CRC bits: {crc}")
        bits_with_crc = np.concatenate((bits[:self.K - self.crc_size], crc))
        print(f"[PolarCode] bits_with_crc (last 16): {bits_with_crc[-16:]}")
        u[idx] = bits_with_crc
        print(f"[PolarCode] u (first 32): {u[:32]}")
        encoded = self._polar_transform(u)
        print(f"[PolarCode] Encoded output (first 32): {encoded[:32]}")
        return encoded

    def decode(self, llr: np.ndarray):
        return self._sc_list_decode(llr)

    def _sc_list_decode(self, llr):
        # ===== ADD THIS DEBUG BLOCK =====
        print(f"[DEBUG] LLR for last 10 positions: {llr[-10:]}")
        print(f"[DEBUG] LLR > 0 count: {np.sum(llr > 0)}/{len(llr)}")

        # Check info positions
        info_pos = np.where(~self.frozen)[0]
        print(f"[DEBUG] Info positions count: {len(info_pos)}")
        print(f"[DEBUG] Last 8 info positions: {info_pos[-8:]}")
        print(f"[DEBUG] LLR at last 8 info positions: {llr[info_pos[-8:]] if len(info_pos) >= 8 else 'N/A'}")
        # ===== END DEBUG BLOCK =====

        # hard decision: +LLR → 1, –LLR → 0
        hard_bits = (llr > 0).astype(np.uint8)

        # ===== ADD THIS DEBUG BLOCK =====
        print(f"[DEBUG] Hard bits for last 10 positions: {hard_bits[-10:]}")
        print(
            f"[DEBUG] Hard bits at last 8 info positions: {hard_bits[info_pos[-8:]] if len(info_pos) >= 8 else 'N/A'}")
        # ===== END DEBUG BLOCK =====

        # F^{⊗n} is involutory over GF(2) ⇒ applying it again inverts it
        u_hat = self._polar_transform(hard_bits)
        data = u_hat[~self.frozen]  # drop frozen positions

        # ===== ADD THIS DEBUG BLOCK =====
        print(f"[DEBUG] u_hat at last 8 info positions: {u_hat[info_pos[-8:]] if len(info_pos) >= 8 else 'N/A'}")
        print(f"[DEBUG] Extracted data (last 8): {data[-8:]}")
        # ===== END DEBUG BLOCK =====

        info, crc = data[:-self.crc_size], data[-self.crc_size:]
        print(f"[DECODER] hard_bits (first 32): {hard_bits[:32]}")
        print(f"[DECODER] u_hat (first 32): {u_hat[:32]}")
        print(f"[DECODER] info bits (first 32): {info[:32]}")
        print(f"[DECODER] crc bits: {crc}")
        print(f"[DECODER] CRC from info: {self._crc8(info)}")
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
