import numpy as np

from rtwm.utils import (
    BAND_PLAN,
    StreamPRNG,
    choose_band,
    db_to_lin,
    lin_to_db,
    mseq_63,
    pn_bits,
    resample_to,
)


def test_mseq_is_maximal_length():
    s = mseq_63()
    assert s.size == 63
    assert set(np.unique(s)).issubset({0, 1})
    assert int(s.sum()) == 32  # 32 ones, 31 zeros

    x = 2 * s.astype(int) - 1
    ac = np.array([int(np.sum(x * np.roll(x, k))) for k in range(1, 63)])
    assert ac.max() == -1 and ac.min() == -1  # ideal 2-level autocorrelation


def test_choose_band_in_plan_and_keyed():
    for ctr in range(50):
        assert choose_band(b"\x01" * 32, ctr) in BAND_PLAN
    seq_a = [choose_band(b"\x01" * 32, c) for c in range(64)]
    seq_b = [choose_band(b"\x02" * 32, c) for c in range(64)]
    assert seq_a != seq_b


def test_db_helpers_roundtrip():
    for db in (-40.0, -20.0, -6.0, 0.0):
        assert abs(lin_to_db(db_to_lin(db)) - db) < 1e-6


def test_resample_to():
    x = np.sin(2 * np.pi * 1_000 * np.arange(48_000) / 48_000).astype(np.float32)
    y, fs = resample_to(16_000, x, 48_000)
    assert fs == 16_000
    assert abs(y.size - x.size // 3) <= 2

    y2, fs2 = resample_to(48_000, x, 48_000)
    assert fs2 == 48_000 and y2 is x  # identity path is a no-op


def test_stream_prng_matches_pn_bits():
    prng = StreamPRNG(b"\x07" * 32)
    bits = pn_bits(prng, 3, 24)
    raw = prng.bytes(3, 3)
    assert np.array_equal(bits, np.unpackbits(np.frombuffer(raw, dtype="u1")))
