import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector

FS, SECS = 48_000, 3

def test_too_short_clip_returns_false():
    key = b"\xDD"*32
    speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
    wm = WatermarkEmbedder(key).process(speech)
    short = wm[:FS*2]                         # <3 s
    assert WatermarkDetector(key).verify(short, FS) is False
