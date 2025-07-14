import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector

FS   = 48_000
SECS = 3

def test_tx_rx_end_to_end():
    key = b"\xAA" * 16
    speech = (np.random.randn(FS * SECS) * 0.05).astype(np.float32)
    tx = WatermarkEmbedder(key)
    watermarked = tx.process(speech)                # single-chunk embed
    rx = WatermarkDetector(key)
    assert rx.verify(watermarked) is True, "watermark not recovered"

from rtwm.utils import lin_to_db

def test_watermark_strength_within_bounds():
    key = b"\x33" * 16
    tx = WatermarkEmbedder(key)
    speech = (np.random.randn(FS * SECS) * 0.1).astype(np.float32)
    wm = tx.process(speech)
    wm_only = wm - speech
    ratio = np.sqrt(np.mean(wm_only**2)) / np.sqrt(np.mean(speech**2))
    db = lin_to_db(ratio)
    assert -40 < db < -10, f"unexpected watermark dB: {db:.2f}"
