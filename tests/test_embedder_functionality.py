import numpy as np
import pytest

from rtwm.embedder  import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector

@pytest.fixture
def key():
    return b"\xAA" * 32

@pytest.fixture
def params():
    # use default TxParams
    return TxParams()

def test_chips_properties(key, params):
    """
    Each frame of chips must be:
      • exactly (len(preamble)+N) samples long
      • zero‐mean
      • unit‐RMS after normalization
    """
    tx = WatermarkEmbedder(key, params)
    chips = tx._make_frame_chips()

    # expected frame length = preamble (63) + N (default 1024)
    expected_len = len(params.preamble) + params.N
    assert chips.shape[0] == expected_len, f"got {chips.shape[0]}, want {expected_len}"

    # statistical invariants
    mean = np.mean(chips)
    rms  = np.sqrt(np.mean(chips**2))
    assert abs(mean) < 1e-3, f"Mean too large: {mean}"
    assert abs(rms - 1.0) < 1e-3,  f"RMS not normalized: {rms}"

def test_tx_rx_roundtrip(key, params):
    """
    A perfect, noise-free frame from the embedder must pass the detector’s
    raw‐frame verify (i.e. be fully recoverable).
    """
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = 0
    chips = tx._make_frame_chips()

    detector = WatermarkDetector(key)
    assert detector.verify(chips, params.fs), "Detector failed to recover payload"
    # assert detector.verify_raw_frame(chips), "Detector failed to recover payload"

def test_process_function_injects_chips(key, params):
    """
    The high-level `process(samples)` call should:
      • preserve sample length
      • actually inject nonzero watermark energy
      • handle buffer rollover correctly across multiple calls
    """
    fsig = np.zeros(2000, dtype=np.float32)
    tx = WatermarkEmbedder(key, params)

    out1 = tx.process(fsig)
    assert out1.shape == fsig.shape
    # watermark added => not all zeros
    assert not np.allclose(out1, fsig)

    # a second call should also work
    out2 = tx.process(fsig)
    assert out2.shape == fsig.shape
    assert not np.allclose(out2, fsig)

def test_payload_uniqueness_and_decrypt(key):
    """
    _build_payload() should always produce 55-byte ciphertexts that
    decrypt back to the right counter and never repeat the nonce.
    """
    from rtwm.crypto import SecureChannel
    tx = WatermarkEmbedder(key, TxParams())
    seen = set()
    for ctr in range(4):
        tx.frame_ctr = ctr
        blob = tx._build_payload()
        assert isinstance(blob, bytes) and len(blob) == 55

        # decrypt and check counter
        plain = SecureChannel(key).open(blob)
        assert plain.startswith(b"ESAL")
        assert int.from_bytes(plain[4:8], "big") == ctr

        # uniqueness
        assert blob not in seen
        seen.add(blob)
