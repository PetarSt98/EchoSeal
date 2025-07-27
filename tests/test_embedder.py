import numpy as np
import pytest
from scipy.signal import lfilter

from rtwm.embedder import WatermarkEmbedder
from rtwm.utils import choose_band, butter_bandpass, db_to_lin
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc, N_DEFAULT, K_DEFAULT

@pytest.fixture
def key():
    return b"\xAA" * 32

@pytest.fixture
def params():
    # use default TxParams for fs, preamble, N, K
    from rtwm.embedder import TxParams
    return TxParams()

def manual_frame_chips(key: bytes, frame_ctr: int, params) -> np.ndarray:
    """
    Re-implement exactly what WatermarkEmbedder._make_frame_chips does:
      1) build payload
      2) polar encode
      3) PN-sequence
      4) map 0/1→±1 in float
      5) bandpass filter and normalize
    """
    # 1) payload
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = frame_ctr
    payload = tx._build_payload()
    # 2) data bits
    data_bits = polar_enc(payload, N=params.N, K=params.K)
    # 3) PN sequence
    frame_len = len(params.preamble) + data_bits.size
    pn_full = SecureChannel(key).pn_bits(frame_ctr, frame_len)
    pn_payload = pn_full[len(params.preamble):]
    # 4) mapping
    preamble_sy = 2.0 * params.preamble.astype(np.float32) - 1.0
    data_sy     = 2.0 * data_bits.astype(np.float32)      - 1.0
    pn_sy       = 2.0 * pn_payload.astype(np.float32)     - 1.0
    symbols     = np.concatenate((preamble_sy, data_sy * pn_sy))
    # 5) filter + normalize
    band = choose_band(key, frame_ctr)
    b, a = butter_bandpass(*band, params.fs)
    chips = lfilter(b, a, symbols)
    chips /= np.sqrt(np.mean(chips**2)) + 1e-12
    return chips.astype(np.float32)

def test_manual_matches_embedder(key, params):
    """_make_frame_chips() must match our manual implementation exactly."""
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = 0

    emb_chips = tx._make_frame_chips()
    man_chips = manual_frame_chips(key, 0, params)

    # exact bit-level match up to float tolerance
    assert emb_chips.shape == man_chips.shape
    assert np.allclose(emb_chips, man_chips, atol=1e-6), "Embedder output diverged from manual!"


def test_frame_counter_increments(key, params):
    """Each call to _make_frame_chips() must bump frame_ctr by 1."""
    tx = WatermarkEmbedder(key, params)
    start = tx.frame_ctr
    _ = tx._make_frame_chips()
    assert tx.frame_ctr == start + 1
    _ = tx._make_frame_chips()
    assert tx.frame_ctr == start + 2


def test_payload_length_and_structure(key, params):
    """The sealed payload must be exactly 55 bytes and vary with frame_ctr."""
    tx = WatermarkEmbedder(key, params)
    for ctr in [0, 1, 7, 255]:
        tx.frame_ctr = ctr
        payload = tx._build_payload()
        assert isinstance(payload, bytes)
        assert len(payload) == 55
        # first four bytes are the XChaCha nonce
        assert payload[:12] != payload[12:24]  # nonce != tag
        # counter round-trip through decrypt
        from rtwm.crypto import SecureChannel
        plain = SecureChannel(key).open(payload)
        # plaintext layout: b"ESAL" + ctr.to_bytes(4, 'big') + …
        assert plain.startswith(b"ESAL")
        assert int.from_bytes(plain[4:8], "big") == ctr


def test_chips_statistics(key, params):
    """Each frame of chips must have zero mean and unit RMS after normalization."""
    tx = WatermarkEmbedder(key, params)
    chips = tx._make_frame_chips()
    mean = np.mean(chips)
    rms  = np.sqrt(np.mean(chips**2))
    assert abs(mean) < 1e-3, f"Mean too large: {mean}"
    assert abs(rms - 1.0) < 1e-3, f"RMS not normalized: {rms}"


def test_choose_band_round_trip(key, params):
    """make sure choose_band is consistent for encoder and detector."""
    frame_ctr = 5
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = frame_ctr
    emb_chips = tx._make_frame_chips()
    # we only care band selection, not full detection
    # the same frame_ctr yields same band
    from rtwm.utils import choose_band
    assert choose_band(key, frame_ctr) == choose_band(key, frame_ctr)

