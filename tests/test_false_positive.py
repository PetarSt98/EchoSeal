import numpy as np
from scipy.signal import butter, lfilter
from rtwm.detector import WatermarkDetector
from rtwm.embedder import WatermarkEmbedder

FS, SECS = 48_000, 3

def test_random_audio_not_authenticated():
    key = b"\xBB"*32
    noise = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
    assert WatermarkDetector(key).verify(noise, FS) is False

def test_lowpass_removal_fails():
    key = b"\xCC"*32
    speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
    wm = WatermarkEmbedder(key).process(speech)
    b, a = butter(4, 0.35)                   # LPF below watermark bands
    filtered = lfilter(b, a, wm)
    assert WatermarkDetector(key).verify(filtered, FS) is False
