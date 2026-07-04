"""
Fast TX -> RX round-trip tests (digital loopback, speech-shaped host noise).

The whole file must run in seconds: happy paths decode through the polar
hard-decision fast path, and rejection paths are gated before any list
decoding happens.  Shared fixtures live in conftest.py.
"""
import numpy as np

from rtwm import frame
from rtwm.detector import WatermarkDetector
from rtwm.utils import resample_to

from conftest import FS, KEY


def test_loopback_verify(marked_10s):
    assert WatermarkDetector(KEY).verify(marked_10s, FS) is True


def test_scan_reports_consecutive_frames(marked_10s):
    hits = WatermarkDetector(KEY).scan(marked_10s, FS)
    assert len(hits) >= 8  # 10 s contain ~12 frames

    ctrs = [h.ctr for h in hits]
    starts = np.array([h.start for h in hits])
    assert ctrs == sorted(ctrs)
    assert set(ctrs).issubset(set(range(13)))
    # Frames are back-to-back: consecutive hits must be ~1 frame apart.
    spacing = np.diff(starts) / frame.frame_samples()
    assert np.all(np.abs(spacing - np.round(spacing)) < 0.01)


def test_excerpt_from_middle_verifies(marked_10s):
    """Core requirement: a ~5 s excerpt cut anywhere must verify on its own."""
    excerpt = marked_10s[int(3.123 * FS) : int(8.35 * FS)]
    report = WatermarkDetector(KEY).analyze(excerpt, FS)

    assert report.verdict == "authentic"
    assert len(report.hits) >= 4
    # Counters are recovered from the frame headers, not from file position.
    assert min(h.ctr for h in report.hits) >= 3


def test_polarity_inverted_recording_verifies(marked_10s):
    excerpt = -marked_10s[: int(4.0 * FS)]
    assert WatermarkDetector(KEY).verify(excerpt, FS) is True


def test_resampled_recording_verifies(marked_10s):
    """Detector must handle non-48k inputs (e.g. 44.1 kHz recordings)."""
    excerpt = marked_10s[: int(5.0 * FS)]
    resampled, fs = resample_to(44_100, excerpt.astype(np.float64), FS)
    assert WatermarkDetector(KEY).verify(resampled, fs) is True


def test_unwatermarked_audio_is_rejected(make_host):
    noise = make_host(4.0, seed=99)
    assert WatermarkDetector(KEY).verify(noise, FS) is False


def test_wrong_key_is_rejected(marked_10s):
    excerpt = marked_10s[: int(4.0 * FS)]
    assert WatermarkDetector(b"\x33" * 32).verify(excerpt, FS) is False


def test_clip_shorter_than_one_frame_is_rejected(marked_10s):
    short = marked_10s[: frame.frame_samples() // 2]
    assert WatermarkDetector(KEY).verify(short, FS) is False


def test_stereo_input_is_downmixed(marked_10s):
    excerpt = marked_10s[: int(4.0 * FS)]
    stereo = np.stack([excerpt, excerpt], axis=1)
    assert WatermarkDetector(KEY).verify(stereo, FS) is True
