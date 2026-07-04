"""
End-to-end scenario tests: the "real picture" of the system.

Synthetic speech-like audio is streamed through the embedder in real-time
block sizes (1024 samples, as the sound card would deliver), then handed to
the offline detector, which must detect -> decode -> decrypt -> validate.

Scenarios:
    1. clean recording                       -> authentic
    2. words cut out (malicious edit)        -> tampered (timeline mismatch)
    3. passage replaced by AI-generated
       audio carrying a forged watermark     -> tampered (unverified span)
    4. same-length word swap (no seq. gap)   -> tampered (absent or fake WM)
    5. word inserted (recording longer)      -> tampered (position mismatch)
    6. tiny same-length swap                 -> tampered (waveform / gap check)
"""
import numpy as np
import pytest
from scipy.signal import butter, lfilter, sawtooth

from rtwm.detector import WatermarkDetector
from rtwm.embedder import WatermarkEmbedder

from conftest import FS, KEY

ATTACKER_KEY = b"\x66" * 32
BLOCK = 1024  # sound-card sized processing blocks


def synth_speech(seconds: float, seed: int = 0, f0: float = 120.0) -> np.ndarray:
    """Speech-like signal: glottal sawtooth through formant resonators, with
    syllable/word amplitude modulation, sibilant bursts and room tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS

    # Voiced source: sawtooth with vibrato, shaped by three formants.
    inst_f = f0 * (1.0 + 0.04 * np.sin(2 * np.pi * 4.7 * t))
    src = sawtooth(2 * np.pi * np.cumsum(inst_f) / FS)
    voiced = np.zeros(n)
    for fcen, bw, w in [(700, 300, 1.0), (1200, 350, 0.7), (2600, 500, 0.4)]:
        b, a = butter(2, [(fcen - bw / 2) / (FS / 2), (fcen + bw / 2) / (FS / 2)], "band")
        voiced += w * lfilter(b, a, src)

    # Syllable (~3 Hz) and word (~0.8 Hz) envelopes; never fully silent.
    syllable = 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.3 * t))
    word = np.clip(1.2 * np.sin(2 * np.pi * 0.8 * t + 0.7), 0.3, 1.0)

    # Sibilant bursts (high-pass noise) in the word gaps, plus room tone.
    b, a = butter(4, 5_000 / (FS / 2), "high")
    sibilant = lfilter(b, a, rng.standard_normal(n))
    sib_env = np.clip(-np.sin(2 * np.pi * 0.8 * t + 0.7), 0.0, 1.0) ** 3
    b, a = butter(2, 1_000 / (FS / 2), "low")
    room = lfilter(b, a, rng.standard_normal(n))

    speech = voiced * syllable * word + 0.15 * sibilant * sib_env + 0.2 * room
    speech = speech / (np.sqrt(np.mean(speech**2)) + 1e-12) * 0.1
    return speech.astype(np.float32)


def stream_embed(key: bytes, audio: np.ndarray) -> np.ndarray:
    """Feed audio through the embedder exactly like the live audio loop."""
    tx = WatermarkEmbedder(key)
    out = [tx.process(audio[i : i + BLOCK]) for i in range(0, audio.size, BLOCK)]
    return np.concatenate(out)


# ────────────────────────── 1. clean recording ─────────────────────────────
def test_e2e_clean_recording_is_authentic():
    speech = synth_speech(8.0, seed=1)
    marked = stream_embed(KEY, speech)

    # Watermark stays well below the speech (masking-aware level policy).
    wm = marked - speech
    level_db = 20 * np.log10(np.sqrt(np.mean(wm**2)) / np.sqrt(np.mean(speech**2)))
    assert -24.0 < level_db < -14.0

    report = WatermarkDetector(KEY).analyze(marked, FS)
    assert report.verdict == "authentic"
    assert len(report.hits) >= 6          # ~9 frames fit into 8 s
    assert report.coverage >= 0.7
    assert not report.unverified_spans


# ─────────────────────── 2. words cut out (edit) ───────────────────────────
def test_e2e_cutting_words_is_flagged_tampered():
    """Someone removes 0.8 s of speech to change the meaning: the counters of
    the surviving frames no longer match their positions."""
    marked = stream_embed(KEY, synth_speech(8.0, seed=2))
    edited = np.concatenate((marked[: int(3.2 * FS)], marked[int(4.0 * FS) :]))

    report = WatermarkDetector(KEY).analyze(edited, FS)
    assert report.verdict == "tampered"
    assert any("missing" in issue for issue in report.issues)


# ─────────────── 3. AI-generated replacement (same length) ─────────────────
def test_e2e_ai_replacement_is_flagged_tampered():
    """A 2.5 s passage is replaced by AI-generated 'speech' of identical
    length, watermarked by the attacker with their own key.  The timeline
    stays intact, so the evidence is the hole: no frame in the replaced span
    verifies under our key, while everything around it does."""
    marked = stream_embed(KEY, synth_speech(10.0, seed=3))

    fake_voice = synth_speech(2.5, seed=99, f0=95.0)      # "different speaker"
    fake_marked = stream_embed(ATTACKER_KEY, fake_voice)  # forged watermark

    lo, hi = int(3.0 * FS), int(3.0 * FS) + fake_marked.size
    replaced = marked.copy()
    replaced[lo:hi] = fake_marked

    report = WatermarkDetector(KEY).analyze(replaced, FS)
    assert report.verdict == "tampered"
    assert any("AI-generated" in issue or "replaced" in issue for issue in report.issues)

    # The reported unverified span must overlap the actual replacement window.
    assert any(t0 < 5.5 and t1 > 3.0 and (t1 - t0) >= 1.0
               for t0, t1 in report.unverified_spans)
    # And the forged watermark must not validate as ours anywhere.
    assert all(h.nonce == report.hits[0].nonce for h in report.hits)


# ──────────── 4. same-length swap, no sequence gap ─────────────────────────
@pytest.mark.parametrize("forged", [False, True])
def test_e2e_same_length_word_swap_no_sequence_gap(forged):
    """Word replaced in-place: total length unchanged, counters unchanged.

    There is no sequence-number gap and no timeline stretch — only a local
    region where our watermark is absent (plain speech) or forged (attacker
    key; AEAD never validates under ours).  Detection must come from the
    watermark hole / local-edit path, not from counter-vs-position."""
    marked = stream_embed(KEY, synth_speech(8.0, seed=6))
    word = synth_speech(0.35, seed=77, f0=160.0)
    if forged:
        word = stream_embed(ATTACKER_KEY, word)

    lo = int(3.0 * FS)
    lo_s, hi_s = lo / FS, (lo + word.size) / FS
    swapped = marked.copy()
    swapped[lo : lo + word.size] = word
    assert swapped.size == marked.size

    report = WatermarkDetector(KEY).analyze(swapped, FS)
    assert report.verdict == "tampered"

    # Same-length edit: position-vs-counter checks must not fire.
    assert not any("inserted" in issue for issue in report.issues)
    assert not any("audio missing" in issue for issue in report.issues)

    # Hole or local-edit evidence must overlap the pasted word.
    assert any(t0 < hi_s and t1 > lo_s for t0, t1 in report.unverified_spans)
    assert any(
        "replaced" in issue or "local edit" in issue for issue in report.issues
    )

    if forged:
        assert all(h.nonce == report.hits[0].nonce for h in report.hits)


# ──────────── 5. word INSERTED (recording gets longer) ─────────────────────
@pytest.mark.parametrize("forged_mark", [False, True])
def test_e2e_inserted_word_is_flagged_tampered(forged_mark):
    """Someone inserts a word the speaker never said.  No original chunk is
    missing and none is renumbered — but every chunk after the insert now sits
    later than its counter allows.  Must hold whether the inserted audio
    carries no watermark at all or a forged one (wrong key never validates)."""
    marked = stream_embed(KEY, synth_speech(8.0, seed=6))
    word = synth_speech(0.4, seed=77, f0=160.0)
    if forged_mark:
        word = stream_embed(ATTACKER_KEY, word)

    cut = int(3.5 * FS)  # mid-speech, not aligned to any chunk boundary
    longer = np.concatenate((marked[:cut], word, marked[cut:]))

    report = WatermarkDetector(KEY).analyze(longer, FS)
    assert report.verdict == "tampered"
    assert any("inserted" in issue for issue in report.issues)


# ──────────── 6. single AI word swapped, same length ───────────────────────
@pytest.mark.parametrize("word_seconds", [0.06, 0.18])
def test_e2e_single_word_ai_swap_is_flagged_tampered(word_seconds):
    """One short word replaced by AI audio of identical length: the timeline
    stays intact.  A tiny swap leaves the frame decodable and is localised by
    the waveform-integrity check; a bigger one kills the frame and is caught
    by the frame-gap check.  Both must yield 'tampered'."""
    marked = stream_embed(KEY, synth_speech(8.0, seed=4))
    word = synth_speech(word_seconds, seed=55, f0=180.0)  # "different voice"

    lo = int(2.0 * FS)  # inside frame 2 (1.62 .. 2.43 s)
    swapped = marked.copy()
    swapped[lo : lo + word.size] = word

    report = WatermarkDetector(KEY).analyze(swapped, FS)
    assert report.verdict == "tampered"
    # The flagged span must overlap the edit window.
    hi_t = 2.0 + word_seconds
    assert any(t0 < hi_t + 0.1 and t1 > 1.95 for t0, t1 in report.unverified_spans)
