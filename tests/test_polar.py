import numpy as np
import pytest

from rtwm.fastpolar import PolarCode
from rtwm.polar_fast import (
    K_DEFAULT,
    N_DEFAULT,
    PAYLOAD_BYTES,
    decode,
    encode,
)


def _awgn_llr(codeword: np.ndarray, sigma: float, rng) -> np.ndarray:
    """BPSK over AWGN; LLR convention log(P1/P0) => llr = 2r/sigma^2."""
    tx = 2.0 * codeword.astype(np.float64) - 1.0
    rx = tx + rng.normal(0.0, sigma, tx.size)
    return 2.0 * rx / sigma**2


# ────────────────────────────── construction ──────────────────────────────
def test_info_set_uses_most_reliable_channels():
    """Regression: info bits belong on the *most* reliable channels.

    Channel 0 is always the least reliable and channel N-1 the most reliable,
    so index 0 must be frozen and index N-1 must carry data.
    """
    pc = PolarCode(N_DEFAULT, K_DEFAULT)
    assert pc.frozen[0]
    assert not pc.frozen[N_DEFAULT - 1]
    assert int((~pc.frozen).sum()) == K_DEFAULT


def test_polar_transform_is_involution():
    rng = np.random.default_rng(0)
    u = rng.integers(0, 2, 1024, dtype=np.uint8)
    once = PolarCode._polar_transform(u)
    assert np.array_equal(PolarCode._polar_transform(once), u)


def test_crc8_known_answer():
    """CRC-8 (poly 0x07, init 0) of b'123456789' is the standard 0xF4."""
    pc = PolarCode(N_DEFAULT, K_DEFAULT)
    bits = np.unpackbits(np.frombuffer(b"123456789", dtype=np.uint8))
    crc = pc._crc8(bits)
    assert int(np.packbits(crc)[0]) == 0xF4


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        PolarCode(1000, 448)          # not a power of two
    with pytest.raises(ValueError):
        PolarCode(2048, 448)          # exceeds reliability table
    with pytest.raises(ValueError):
        PolarCode(1024, 4)            # K <= crc_size
    with pytest.raises(ValueError):
        PolarCode(1024, 448, crc_size=16)  # only CRC-8 supported


# ────────────────────────────── round-trips ───────────────────────────────
def test_clean_roundtrip():
    rng = np.random.default_rng(1)
    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8)
    info = rng.integers(0, 2, pc.K - pc.crc_size, dtype=np.uint8)
    cw = pc.encode(info)

    assert cw.shape == (N_DEFAULT,)
    llr = 20.0 * (cw.astype(np.float64) - 0.5)
    bits, ok = pc.decode(llr)
    assert ok
    assert np.array_equal(bits, info)


def test_awgn_roundtrip_at_realistic_snr():
    """SCL-8 must recover the payload at ~2.5 dB per-chip SNR (sigma=0.75).

    This is the operating point that exposed the frozen-set direction bug:
    with info bits on the wrong channels the decoder fails 100% here.
    """
    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8)
    for seed in (10, 11, 12):
        rng = np.random.default_rng(seed)
        info = rng.integers(0, 2, pc.K - pc.crc_size, dtype=np.uint8)
        llr = _awgn_llr(pc.encode(info), 0.75, rng)
        bits, ok = pc.decode(llr)
        assert ok, f"decode failed at seed {seed}"
        assert np.array_equal(bits, info)


def test_hopeless_snr_reports_failure():
    rng = np.random.default_rng(2)
    payload = rng.bytes(PAYLOAD_BYTES)
    llr = _awgn_llr(encode(payload), 2.5, rng)  # ~ -8 dB per-chip SNR
    assert decode(llr, validator=lambda b: b == payload) is None


def test_smaller_block_length():
    """Nested reliability sequence supports any power-of-two N <= 1024."""
    rng = np.random.default_rng(3)
    pc = PolarCode(256, 112, list_size=8)
    info = rng.integers(0, 2, pc.K - pc.crc_size, dtype=np.uint8)
    llr = _awgn_llr(pc.encode(info), 0.5, rng)
    bits, ok = pc.decode(llr)
    assert ok
    assert np.array_equal(bits, info)


# ─────────────────────────── byte-level wrapper ───────────────────────────
def test_wrapper_roundtrip_with_noise():
    rng = np.random.default_rng(4)
    payload = rng.bytes(PAYLOAD_BYTES)
    llr = _awgn_llr(encode(payload), 0.75, rng)
    recovered, ok = decode(llr, return_ok=True)
    assert ok
    assert recovered == payload


def test_encode_is_deterministic():
    payload = bytes(range(PAYLOAD_BYTES))
    assert np.array_equal(encode(payload), encode(payload))


def test_validator_arbitrates_between_candidates():
    """The detector uses AEAD-open as validator: it must gate `ok`."""
    payload = bytes(PAYLOAD_BYTES)
    llr = 20.0 * (encode(payload).astype(np.float64) - 0.5)

    assert decode(llr, validator=lambda b: False) is None
    assert decode(llr, validator=lambda b: b == payload) == payload


def test_wrapper_rejects_bad_lengths():
    with pytest.raises(ValueError):
        encode(b"short")
    with pytest.raises(ValueError):
        decode(np.zeros(100))
