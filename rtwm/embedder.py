"""
Real-time spread-spectrum watermark embedder.

For every audio block handed to :meth:`WatermarkEmbedder.process` the embedder
generates watermark frames on demand and mixes them into the host signal:

    payload  →  seal (ChaCha20-Poly1305)  →  polar encode  →
    [ MLS preamble | counter header | PN-scrambled payload ]  →
    BPSK @ 1.5 kBd on a per-frame hop-band carrier  →  unit-RMS waveform

Frames repeat back-to-back (~0.81 s each at 48 kHz), so any few-second excerpt
of the output contains several complete, independently verifiable frames.

Level policy (per block, masking-aware): the watermark sits at
``target_rel_db`` below the host's total RMS, but is raised to match the
host's energy *inside the current hop band* when that is higher — that is
exactly when the band provides masking, and it is what keeps frames decodable
during sibilance/music that would otherwise drown the band.  The level is
capped at ``ceiling_rel_db`` below total RMS, clip-limited against
``MIX_HEADROOM``, and gated in near-silence.  Frame generation costs ~1 ms per
0.81 s frame: comfortably real-time.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from rtwm import frame
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc, K_DEFAULT, N_DEFAULT
from rtwm.utils import BAND_PLAN, butter_bandpass, choose_band, db_to_lin

EPS = 1e-12
MIN_RMS_SILENCE = 1e-4   # ~ -80 dBFS: gate the watermark in near-silence
MIX_HEADROOM = 0.98      # never let host + watermark exceed this peak


@dataclass(slots=True)
class TxParams:
    fs: int = 48_000
    sps: int = frame.SPS
    target_rel_db: float = -20.0   # baseline: WM vs total host RMS
    ceiling_rel_db: float = -12.0  # audibility cap: never louder than this
    N: int = N_DEFAULT
    K: int = K_DEFAULT


class WatermarkEmbedder:
    def __init__(self, key32: bytes, params: TxParams | None = None) -> None:
        self.p = params or TxParams()
        if self.p.N != frame.PAYLOAD_BITS:
            raise ValueError(f"N must equal frame.PAYLOAD_BITS ({frame.PAYLOAD_BITS})")
        self.sec = SecureChannel(key32)
        self.frame_ctr = 0
        self._chip_buf = np.empty(0, dtype=np.float32)
        self._band_buf = np.empty(0, dtype=np.uint8)  # hop-band index per chip
        self._band_filters = {
            band: butter_bandpass(*band, self.p.fs, order=4) for band in BAND_PLAN
        }
        self._session_nonce = secrets.token_bytes(8)

    # ------------------------------------------------------------------ API
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Mix watermark into `samples` and return the result."""
        x = samples.astype(np.float32, copy=False)
        n = x.size

        # Keep generating frames even through silence so the frame counter
        # stays aligned with wall-clock time.
        while self._chip_buf.size < n:
            band_idx = BAND_PLAN.index(choose_band(self.sec.band_key, self.frame_ctr))
            chips = self._make_frame_chips()
            self._chip_buf = np.concatenate((self._chip_buf, chips))
            self._band_buf = np.concatenate(
                (self._band_buf, np.full(chips.size, band_idx, dtype=np.uint8))
            )
            self.frame_ctr = (self.frame_ctr + 1) % (1 << 32)

        chips = self._chip_buf[:n]
        self._chip_buf = self._chip_buf[n:]
        band_ids = self._band_buf[:n]
        self._band_buf = self._band_buf[n:]

        in_rms = float(np.sqrt(np.mean(x * x)) + EPS)
        if in_rms < MIN_RMS_SILENCE:
            # No host signal to mask the watermark -> stay silent.
            return x.copy()

        # Masking-aware level (chips are unit-RMS): baseline `target_rel_db`
        # below total host RMS, raised towards the host's in-band RMS when the
        # band is busy (busy band == masking available == watermark would
        # otherwise drown), capped at `ceiling_rel_db` below total RMS.
        band = BAND_PLAN[int(np.bincount(band_ids).argmax())]
        b, a = self._band_filters[band]
        band_rms = float(np.sqrt(np.mean(lfilter(b, a, x.astype(np.float64)) ** 2)))

        base = db_to_lin(self.p.target_rel_db) * in_rms
        ceiling = db_to_lin(self.p.ceiling_rel_db) * in_rms
        scale = min(max(base, 0.8 * band_rms), ceiling)

        # Clip-safe headroom limiter.
        headroom = max(0.0, MIX_HEADROOM - float(np.max(np.abs(x))))
        peak = float(np.max(np.abs(chips))) + EPS
        scale = min(scale, headroom / peak)

        return x + chips * scale

    # ------------------------------------------------------------------ internals
    def _make_frame_chips(self) -> np.ndarray:
        """Generate one full frame of unit-RMS watermark samples."""
        band = choose_band(self.sec.band_key, self.frame_ctr)

        # Encrypt payload and FEC-encode it to the polar codeword; the header
        # carries the low counter bits under a fixed PN so the receiver can
        # bootstrap the frame counter from the frame itself.
        payload = self._build_payload()
        data_bits = polar_enc(payload, N=self.p.N, K=self.p.K)
        pn = self.sec.pn_bits(self.frame_ctr, frame.SYMBOLS_PER_FRAME)

        symbols = frame.assemble_symbols(
            self.frame_ctr,
            data_bits,
            self.sec.pn_bits(0, frame.HDR_LEN),
            pn[frame.PRE_LEN + frame.HDR_LEN :],
        )
        chips = frame.modulate(symbols, band, self.p.fs, self.p.sps)

        # Normalise to unit RMS on the steady-state part (skip the filter
        # transient) so the mixer's dB scaling is meaningful.
        steady = chips[4 * self.p.sps :]
        energy = float(np.mean(steady * steady))
        if energy > EPS:
            chips = chips / np.sqrt(energy)

        return chips.astype(np.float32, copy=False)

    def _build_payload(self) -> bytes:
        """Return the 55-byte sealed payload (nonce ‖ ciphertext ‖ tag).

        Plaintext (27 bytes): magic "ESAL" ‖ 32-bit counter ‖ 8-byte session
        nonce ‖ 11 random padding bytes.
        """
        meta = (
            b"ESAL"
            + self.frame_ctr.to_bytes(4, "big")
            + self._session_nonce
            + secrets.token_bytes(11)
        )
        assert len(meta) == 27
        blob = self.sec.seal(meta)
        assert len(blob) == 55
        return blob
