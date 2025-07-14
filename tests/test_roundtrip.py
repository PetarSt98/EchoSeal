import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
from rtwm.utils import lin_to_db

FS, SECS = 48_000, 3

def test_tx_rx_end_to_end():
    key = b"\xAA" * 32
    speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)
    assert WatermarkDetector(key).verify(wm, FS) is True

def test_watermark_power_bounds():
    key = b"\x33"*32
    speech = (np.random.randn(FS*SECS) * 0.1).astype(np.float32)
    wm = WatermarkEmbedder(key).process(speech)
    ratio_db = lin_to_db(np.sqrt(np.mean((wm-speech)**2)) /
                         np.sqrt(np.mean(speech**2)))
    assert -40 < ratio_db < -10
