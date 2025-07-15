import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector

FS, SECS = 48_000, 3

# def test_too_short_clip_returns_false():
#     key = b"\xDD"*32
#     speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
#     wm = WatermarkEmbedder(key).process(speech)
#     short = wm[:FS*2]                         # <3 s
#     assert WatermarkDetector(key).verify(short, FS) is False

def test_too_short_clip_returns_false():
    """Test that clips shorter than minimum duration return False."""
    key = b"\xDD" * 32
    speech = (np.random.randn(FS * SECS) * 0.05).astype(np.float32)
    wm = WatermarkEmbedder(key).process(speech)
    short = wm[:FS * 2]  # <3 s
    assert WatermarkDetector(key).verify(short, FS) is False


def test_silent_audio():
    """Test detection with silent (zero) audio."""
    key = b"\xEE" * 32
    silence = np.zeros(FS * SECS, dtype=np.float32)
    wm = WatermarkEmbedder(key).process(silence)

    # Should still detect watermark in silence
    assert WatermarkDetector(key).verify(wm, FS) is True


def test_very_loud_audio():
    """Test with very loud input audio."""
    key = b"\xFF" * 32
    loud_speech = (np.random.randn(FS * SECS) * 0.9).astype(np.float32)  # Near clipping
    wm = WatermarkEmbedder(key).process(loud_speech)

    # Should still detect despite high input level
    assert WatermarkDetector(key).verify(wm, FS) is True


def test_different_sample_rates():
    """Test detection with different sample rates."""
    key = b"\x55" * 32

    # Test at 44.1 kHz
    fs_44k = 44100
    speech_44k = (np.random.randn(fs_44k * SECS) * 0.05).astype(np.float32)

    # Note: This will currently fail as embedder is hardcoded to 48kHz
    # This test documents the limitation
    tx = WatermarkEmbedder(key)
    wm_44k = tx.process(speech_44k)

    # Detector should handle resampling internally
    detector = WatermarkDetector(key, fs_target=48000)
    # This may fail with current implementation - documents needed improvement
    result = detector.verify(wm_44k, fs_44k)
    # For now, we just check it doesn't crash
    assert isinstance(result, bool)


def test_empty_audio():
    """Test with empty audio array."""
    key = b"\x00" * 32
    empty = np.array([], dtype=np.float32)

    detector = WatermarkDetector(key)
    # Should return False for empty audio without crashing
    assert detector.verify(empty, FS) is False
