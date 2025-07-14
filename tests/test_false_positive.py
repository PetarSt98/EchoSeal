import numpy as np
from rtwm.detector import WatermarkDetector
from scipy.signal import butter, lfilter

FS   = 48_000
SECS = 3

def test_random_audio_does_not_fool_detector():
    key = b"\xBB" * 16
    rx  = WatermarkDetector(key)
    noise = (np.random.randn(48_000 * 3) * 0.05).astype("float32")
    assert rx.verify(noise) is False, "false positive detected"

def test_filtered_watermark_fails():
    key = b"\xCC" * 16
    tx = WatermarkEmbedder(key)
    speech = (np.random.randn(FS * SECS) * 0.05).astype(np.float32)
    wm = tx.process(speech)
    # Low-pass filter to simulate recording device removing ultrasonic content
    b, a = butter(4, 0.35)  # removes > ~8kHz @ 48kHz
    filtered = lfilter(b, a, wm)
    rx = WatermarkDetector(key)
    assert rx.verify(filtered) is False, "filtered audio should fail"

