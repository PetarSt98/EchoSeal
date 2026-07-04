import numpy as np
import pytest

from rtwm import frame
from rtwm.crypto import SecureChannel
from rtwm.embedder import MIX_HEADROOM, TxParams, WatermarkEmbedder
from rtwm.utils import choose_band

EPS = 1e-12


@pytest.fixture
def key():
    return b"\xAA" * 32


@pytest.fixture
def tx(key):
    return WatermarkEmbedder(key)


def test_frame_has_expected_length(tx):
    chips = tx._make_frame_chips()
    assert chips.shape == (frame.frame_samples(tx.p.sps),)
    assert chips.dtype == np.float32


def test_frame_is_unit_rms_steady_state(tx):
    steady = tx._make_frame_chips()[4 * tx.p.sps :]
    assert abs(float(np.sqrt(np.mean(steady**2))) - 1.0) < 5e-3
    assert abs(float(np.mean(steady))) < 5e-3


def test_frame_starts_with_preamble_waveform(tx):
    """The frame prefix must equal the shared preamble template (up to scale)."""
    tx.frame_ctr = 3
    chips = tx._make_frame_chips()
    band = choose_band(tx.sec.band_key, tx.frame_ctr)
    tpl = frame.preamble_waveform(band, tx.p.fs, tx.p.sps)

    prefix = chips[: tpl.size].astype(np.float64)
    ncc = np.dot(prefix, tpl) / (np.linalg.norm(prefix) * np.linalg.norm(tpl) + EPS)
    assert ncc > 0.999


def test_frame_energy_is_confined_to_hop_band(tx):
    """Spectral shaping: most frame energy must sit inside the chosen band."""
    for ctr in range(4):
        tx.frame_ctr = ctr
        chips = tx._make_frame_chips().astype(np.float64)
        lo, hi = choose_band(tx.sec.band_key, ctr)

        spec = np.abs(np.fft.rfft(chips)) ** 2
        freqs = np.fft.rfftfreq(chips.size, 1.0 / tx.p.fs)
        in_band = spec[(freqs >= lo - 500) & (freqs <= hi + 500)].sum()
        assert in_band / spec.sum() > 0.7, f"ctr={ctr}, band=({lo},{hi})"


def test_counter_advances_in_process_not_in_frame_gen(tx):
    start = tx.frame_ctr
    tx._make_frame_chips()
    assert tx.frame_ctr == start

    tx.process(np.full(1000, 0.01, dtype=np.float32))
    assert tx.frame_ctr == start + 1


def test_payload_is_sealed_with_counter(key, tx):
    sc = SecureChannel(key)
    for ctr in (0, 1, 255, 65_535):
        tx.frame_ctr = ctr
        blob = tx._build_payload()
        assert len(blob) == 55
        plain = sc.open(blob)
        assert plain[:4] == b"ESAL"
        assert int.from_bytes(plain[4:8], "big") == ctr


def test_silence_gate_returns_input_unchanged(tx):
    samples = np.full(480, 1e-6, dtype=np.float32)
    out = tx.process(samples.copy())
    assert np.allclose(out, samples, atol=1e-12)


def test_mixer_respects_headroom(tx):
    samples = np.full(4096, 0.97, dtype=np.float32)
    out = tx.process(samples.copy())
    assert float(np.max(np.abs(out))) <= MIX_HEADROOM + 1e-5


def _wm_level_db(tx: WatermarkEmbedder, host: np.ndarray) -> float:
    out = tx.process(host.copy())
    wm = out - host
    return 20 * np.log10(
        (np.sqrt(np.mean(wm**2)) + EPS) / (np.sqrt(np.mean(host**2)) + EPS)
    )


def test_watermark_level_tracks_target_when_band_is_quiet(key):
    """Speech-like host (energy < 3 kHz): hop bands are quiet, so the level
    must sit at the -20 dB baseline."""
    from scipy.signal import butter, lfilter

    params = TxParams(target_rel_db=-20.0)
    tx = WatermarkEmbedder(key, params)
    rng = np.random.default_rng(0)
    b, a = butter(4, 3_000 / 24_000, "low")
    host = lfilter(b, a, rng.standard_normal(96_000))
    host = (host / np.sqrt(np.mean(host**2)) * 0.1).astype(np.float32)

    assert abs(_wm_level_db(tx, host) - params.target_rel_db) < 2.0


def test_watermark_level_rises_with_in_band_masking(key):
    """Broadband host puts energy inside the hop band: the level must rise
    towards the in-band host level (masking) but stay under the ceiling."""
    params = TxParams(target_rel_db=-20.0, ceiling_rel_db=-12.0)
    tx = WatermarkEmbedder(key, params)
    rng = np.random.default_rng(0)
    host = (rng.standard_normal(96_000) * 0.1).astype(np.float32)

    level = _wm_level_db(tx, host)
    assert params.target_rel_db - 1.0 < level <= params.ceiling_rel_db + 1.0
    assert level > params.target_rel_db + 3.0  # actually boosted, not baseline


def test_process_handles_arbitrary_block_sizes(tx):
    """Chip buffering must survive block sizes unrelated to the frame length."""
    rng = np.random.default_rng(1)
    outs = []
    for block in (480, 1024, 333, 96_000):
        host = (rng.standard_normal(block) * 0.05).astype(np.float32)
        outs.append(tx.process(host))
    assert sum(o.size for o in outs) == 480 + 1024 + 333 + 96_000


def test_rejects_wrong_codeword_length(key):
    with pytest.raises(ValueError):
        WatermarkEmbedder(key, TxParams(N=512))
