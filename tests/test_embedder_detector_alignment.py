import types

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy.signal")
from scipy.signal import lfilter

from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector, FRAME_LEN, PRE_L, HDR_L
from rtwm.utils import butter_bandpass, choose_band
from rtwm.polar_fast import encode as polar_encode


TEST_KEY = bytes.fromhex("00" * 32)


def _constant_payload(_: WatermarkEmbedder) -> bytes:
    return bytes(range(55))


def test_embedder_detector_sequences_align():
    embedder = WatermarkEmbedder(TEST_KEY)
    detector = WatermarkDetector(TEST_KEY)

    np.testing.assert_allclose(embedder._preamble_sy, detector._pre_sy)
    np.testing.assert_allclose(embedder._hdr_pn_sy, detector._hdr_pn_sy)

    for ctr in (0, 1, 255, 1024):
        pn_tx = embedder.sec.pn_bits(ctr, PRE_L + HDR_L + embedder.p.N)
        pn_rx = detector.sec.pn_bits(ctr, FRAME_LEN)
        np.testing.assert_array_equal(pn_tx, pn_rx)


def test_embedder_frame_filtering_matches_spec():
    embedder = WatermarkEmbedder(TEST_KEY)
    embedder._build_payload = types.MethodType(_constant_payload, embedder)

    frame_ctr = 5
    embedder.frame_ctr = frame_ctr
    frame = embedder._make_frame_chips()

    assert frame.size == FRAME_LEN

    payload = _constant_payload(embedder)
    data_bits = polar_encode(payload, N=embedder.p.N, K=embedder.p.K)
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0

    ctr_lo16 = np.uint16(frame_ctr & 0xFFFF)
    ctr_bytes = np.array([ctr_lo16 >> 8, ctr_lo16 & 0xFF], dtype=np.uint8)
    hdr_bits = np.unpackbits(ctr_bytes)
    hdr_bits_rep = np.repeat(hdr_bits, 8)
    hdr_bpsk = 2.0 * hdr_bits_rep.astype(np.float32) - 1.0
    hdr_sy = hdr_bpsk * embedder._hdr_pn_sy

    pn_full = embedder.sec.pn_bits(frame_ctr, PRE_L + HDR_L + embedder.p.N)
    pn_payload = pn_full[PRE_L + HDR_L:]
    pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

    spread_payload = data_symbols * pn_symbols
    symbols = np.concatenate((embedder._preamble_sy, hdr_sy, spread_payload))
    np.testing.assert_equal(symbols.size, FRAME_LEN)

    band = choose_band(embedder._band_key, frame_ctr)
    b, a = butter_bandpass(*band, embedder.p.fs, order=4)
    zi_len = max(len(a), len(b)) - 1
    zi0 = np.zeros(zi_len, dtype=np.float32)
    y_pre, zi1 = lfilter(b, a, embedder._preamble_sy, zi=zi0)
    y_rest, _ = lfilter(b, a, np.concatenate((hdr_sy, spread_payload)), zi=zi1)
    expected = np.concatenate((y_pre, y_rest))

    np.testing.assert_allclose(frame, expected, rtol=1e-5, atol=1e-5)
