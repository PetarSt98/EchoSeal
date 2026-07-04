"""
Shared TX/RX frame definition — single source of truth for the air format.

A watermark frame is 1215 BPSK symbols:

    [ 63-symbol MLS preamble | 128-symbol counter header | 1024 coded bits ]

Each symbol lasts ``SPS`` samples and is modulated onto a per-band carrier,
then band-pass shaped with a 4th-order Butterworth (the frequency hop band is
chosen per frame from the keyed band plan).  The carrier frequency is snapped
to a multiple of the symbol rate so every symbol contains an integer number of
carrier cycles: all symbols share one pulse shape and a rectangular
integrate-and-dump demodulator rejects the double-frequency image exactly.

At 48 kHz / SPS=32 the symbol rate is 1.5 kBd, a frame lasts 38 880 samples
(0.81 s) and repeats continuously, so any few-second excerpt of a recording
contains multiple complete, independently verifiable frames.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from rtwm.utils import butter_bandpass, mseq_63

SPS = 32                    # samples per symbol (spreading factor)

PRE_LEN = 63                # MLS preamble symbols (sync + phase reference)
HDR_BITS = 16               # low 16 bits of the frame counter
HDR_REPEAT = 8              # repetition factor per header bit
HDR_LEN = HDR_BITS * HDR_REPEAT
PAYLOAD_BITS = 1024         # polar codeword length N
SYMBOLS_PER_FRAME = PRE_LEN + HDR_LEN + PAYLOAD_BITS  # 1215

PRE_SYMBOLS = 2.0 * mseq_63().astype(np.float64) - 1.0


def frame_samples(sps: int = SPS) -> int:
    """Frame length in audio samples."""
    return SYMBOLS_PER_FRAME * sps


def carrier_freq(band: tuple[int, int], fs: int, sps: int = SPS) -> float:
    """Band-centre carrier snapped to a multiple of the symbol rate."""
    symbol_rate = fs / sps
    return symbol_rate * round((band[0] + band[1]) / 2 / symbol_rate)


def modulate(
    symbols: np.ndarray,
    band: tuple[int, int],
    fs: int,
    sps: int = SPS,
    *,
    phase: float = 0.0,
) -> np.ndarray:
    """BPSK-modulate ±1 symbols into the band (carrier + Butterworth shaping).

    The carrier phase is zero at the first sample, which the receiver relies
    on: after synchronising on the preamble it estimates the one residual
    channel phase from the preamble symbols.  `phase` produces the quadrature
    variant (-pi/2) used by the detector's waveform-integrity check.
    """
    up = np.repeat(np.asarray(symbols, dtype=np.float64), sps)
    n = np.arange(up.size)
    x = up * np.cos(2.0 * np.pi * carrier_freq(band, fs, sps) * n / fs + phase)
    b, a = butter_bandpass(*band, fs, order=4)
    return lfilter(b, a, x)


def preamble_waveform(band: tuple[int, int], fs: int, sps: int = SPS) -> np.ndarray:
    """TX waveform of the preamble alone (identical to a frame's first part)."""
    return modulate(PRE_SYMBOLS, band, fs, sps)


def assemble_symbols(
    ctr: int,
    data_bits: np.ndarray,
    hdr_pn_bits: np.ndarray,
    payload_pn_bits: np.ndarray,
) -> np.ndarray:
    """Build the ±1 symbol vector of one frame (preamble | header | payload).

    Shared by the embedder (transmit) and the detector (waveform-integrity
    reconstruction of decoded frames), so the two can never diverge.
    """
    ctr_lo16 = ctr & 0xFFFF
    hdr_bits = np.unpackbits(np.array([ctr_lo16 >> 8, ctr_lo16 & 0xFF], dtype=np.uint8))
    hdr_bits = np.repeat(hdr_bits, HDR_REPEAT)
    hdr_sym = (2.0 * hdr_bits - 1.0) * (2.0 * hdr_pn_bits.astype(np.float64) - 1.0)

    data_sym = 2.0 * data_bits.astype(np.float64) - 1.0
    payload_sym = data_sym * (2.0 * payload_pn_bits.astype(np.float64) - 1.0)

    return np.concatenate((PRE_SYMBOLS, hdr_sym, payload_sym))
