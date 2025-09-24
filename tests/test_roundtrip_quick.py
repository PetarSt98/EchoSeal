from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector, FRAME_LEN


def test_tx_rx_end_to_end_quick():
    """Decode a single synthesized frame to exercise the detector quickly."""

    key = b"\xAA" * 32

    tx = WatermarkEmbedder(key)
    frame = tx._make_frame_chips()
    assert frame.shape[0] == FRAME_LEN

    rx = WatermarkDetector(key, list_size=32)
    assert rx.verify_raw_frame(frame)
