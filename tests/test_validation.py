"""
Verdict-layer tests: decode -> decrypt -> validate on realistic recordings.

Benign channels (mild noise, phone-codec low-pass, volume changes) must stay
"authentic" — degradation alone is never tampering.  Real tampering (cuts,
splices of sessions, replayed segments) must be flagged with a reason.
"""
import numpy as np
from scipy.signal import firwin, lfilter

from rtwm import frame
from rtwm.detector import WatermarkDetector
from rtwm.embedder import WatermarkEmbedder

from conftest import FS, KEY


def _analyze(audio):
    return WatermarkDetector(KEY).analyze(audio, FS)


# ───────────────────── benign channels stay authentic ─────────────────────
def test_mild_noise_is_still_authentic(marked_10s):
    """Additive noise at the watermark's own power level must not break it."""
    rng = np.random.default_rng(7)
    noisy = marked_10s[: 5 * FS] + rng.standard_normal(5 * FS).astype(np.float32) * 0.01
    report = _analyze(noisy)
    assert report.verdict == "authentic"
    assert len(report.hits) >= 3


def test_phone_codec_lowpass_is_still_authentic(marked_10s):
    """A harsh phone/codec chain low-passes around ~10 kHz.  That attenuates
    the 10–12 and 12–14 kHz hop bands, but 6–8 and 8–10 kHz frames survive —
    missing frames are degradation, not tampering."""
    taps = firwin(501, 10_000, fs=FS)
    clipped = lfilter(taps, 1.0, marked_10s).astype(np.float32)

    report = _analyze(clipped)
    assert report.verdict == "authentic"
    assert len(report.hits) >= 3          # lower hop bands survive
    assert not report.issues


def test_volume_change_is_still_authentic(marked_10s):
    report = _analyze(marked_10s[: 5 * FS] * 0.25)
    assert report.verdict == "authentic"


# ───────────────────────── tampering is detected ──────────────────────────
def test_cut_in_the_middle_is_detected(marked_10s):
    """Removing 1.3 s must show up as a counter/position mismatch."""
    cut = np.concatenate((marked_10s[: 4 * FS], marked_10s[int(5.3 * FS) :]))
    report = _analyze(cut)
    assert report.verdict == "tampered"
    assert any("missing" in issue for issue in report.issues)


def test_surgical_whole_frame_removal_is_detected(marked_10s):
    """Adversarial best case: frames 4-6 removed EXACTLY at frame boundaries.

    Every surviving frame is bit-perfect (nothing corrupted), so only the
    counter-vs-position arithmetic can catch it: frame 7 now sits where
    frame 4 used to start."""
    L = frame.frame_samples()
    cut = np.concatenate((marked_10s[: 4 * L], marked_10s[7 * L :]))

    report = _analyze(cut)
    assert report.verdict == "tampered"
    # ~3 frames = ~2.43 s reported missing between frames 3 and 7.
    assert any("missing" in issue and "frames 3 and 7" in issue
               for issue in report.issues)


def test_insertion_between_frames_is_detected(marked_10s):
    """Foreign audio inserted exactly at a frame boundary (no frame damaged):
    positions now exceed what the counters allow."""
    L = frame.frame_samples()
    rng = np.random.default_rng(5)
    filler = (rng.standard_normal(FS // 2) * 0.05).astype(np.float32)  # 0.5 s
    spliced = np.concatenate((marked_10s[: 4 * L], filler, marked_10s[4 * L :]))

    report = _analyze(spliced)
    assert report.verdict == "tampered"
    assert any("inserted" in issue for issue in report.issues)


def test_splice_of_two_sessions_is_detected(make_host):
    """Concatenating two recordings (same key, different sessions) must fail
    the session-nonce consistency check."""
    a = WatermarkEmbedder(KEY).process(make_host(4.0, seed=1))
    b = WatermarkEmbedder(KEY).process(make_host(4.0, seed=2))
    report = _analyze(np.concatenate((a, b)))
    assert report.verdict == "tampered"
    assert any("nonce" in issue for issue in report.issues)


def test_replayed_segment_is_detected(marked_10s):
    """Doubling a segment repeats frame counters -> replay evidence."""
    excerpt = marked_10s[: int(2.5 * FS)]
    report = _analyze(np.concatenate((excerpt, excerpt)))
    assert report.verdict == "tampered"
    assert any("twice" in issue for issue in report.issues)


# ─────────────────────────── verdict semantics ────────────────────────────
def test_unwatermarked_audio_reports_no_watermark(make_host):
    report = _analyze(make_host(3.0, seed=42))
    assert report.verdict == "no-watermark"
    assert report.hits == [] and report.issues == []


def test_coverage_reflects_found_fraction(marked_10s):
    report = _analyze(marked_10s)
    assert report.verdict == "authentic"
    assert report.coverage > 0.8
