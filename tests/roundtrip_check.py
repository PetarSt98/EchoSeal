"""
Minimal reproduction tests to isolate the exact failure point.
"""
import numpy as np

def test_llr_shift_truncation_debug():
    """
    Build a multi-frame chips stream directly from the embedder.
    Slice exact frame windows (1087 samples each), and inspect
    the detector's LLR meta-logs to quantify tail zeros.
    """
    import numpy as np
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector, FRAME_LEN, N_DEFAULT

    key = b"\xAA"*32
    tx  = WatermarkEmbedder(key)
    # Generate 10 frames worth of filtered chips
    frames = [tx._make_frame_chips() for _ in range(10)]
    stream = np.concatenate(frames).astype(np.float32)

    rx = WatermarkDetector(key)

    tail_zeros = []
    for k in range(8):  # first 8 frames
        start = k * FRAME_LEN
        frame = stream[start:start+FRAME_LEN]
        llr = rx._llr(frame, k)  # triggers new [LLR ALIGN] logs
        assert llr.size == N_DEFAULT
        tz = int(np.sum(np.isclose(llr[-64:], 0.0)))
        tail_zeros.append(tz)

    print("TAIL ZEROS per frame (last 64 LLRs):", tail_zeros)
    # With the current truncation bug, expect many frames to have tz > 0
    assert any(z > 0 for z in tail_zeros), "No tail zeros observed; truncation not reproduced"


def test_preamble_peaks_near_true_boundaries():
    import numpy as np
    from scipy.signal import lfilter, correlate
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector, PRE_L, FRAME_LEN, mseq_63
    from rtwm.utils import butter_bandpass, choose_band

    key = b"\xAA"*32
    tx  = WatermarkEmbedder(key)
    # Build ~30 frames of chips and then pass them through *detector's* bandpass once more
    chips = np.concatenate([tx._make_frame_chips() for _ in range(30)]).astype(np.float32)

    rx = WatermarkDetector(key)
    band = choose_band(getattr(rx.sec, "band_key", key), 0)

    b,a = butter_bandpass(*band, rx.fs_target, order=4)
    y,_ = chips, None  # chips are already filtered by TX; use as-is to keep timing
    # Detector's preamble template:
    pre_sy = 2.0 * mseq_63().astype(np.float32) - 1.0
    tpl = lfilter(b, a, pre_sy)
    tpl = tpl / (np.sqrt(np.sum(tpl*tpl)) + 1e-12)

    L = tpl.size
    y2 = y*y
    e_y = np.sqrt(np.convolve(y2, np.ones(L, dtype=np.float32), mode="valid")) + 1e-12
    corr = correlate(y, tpl, mode="valid") / e_y

    # Pull top 30 peaks
    idx = np.argsort(corr)[-30:]
    peaks = np.sort(idx)

    # Compare to true frame starts (0, FRAME_LEN, 2*FRAME_LEN, ...)
    true_starts = np.arange(0, chips.size - FRAME_LEN + 1, FRAME_LEN)
    nearest = []
    for p in peaks:
        d = np.min(np.abs(true_starts - p))
        nearest.append(int(d))
    print("Distance from peaks to nearest true frame start (samples):", nearest)

    # Expect distances clustered within small O(filter tail) offsets
    assert np.median(nearest) <= 16


def test_quiet_noise_roundtrip_with_alignment_logs():
    import numpy as np
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector

    key = b"\xAA"*32
    FS  = 48000
    SECS = 5

    np.random.seed(52)
    noise = 0.01 * np.random.randn(FS * SECS).astype(np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(noise)

    print(f"Signal power: {np.mean(noise**2):.6f}")
    print(f"Watermark power: {np.mean((wm-noise)**2):.6f}")
    print(f"SNR: {10*np.log10(np.mean(noise**2)/np.mean((wm-noise)**2)):.1f} dB")

    rx = WatermarkDetector(key)
    ok = rx.verify(wm, FS)
    assert ok is True, "Quiet noise roundtrip still failing (see [LLR ALIGN] logs above)"


def test_quiet_noise_roundtrip_with_alignment_logs():
    import numpy as np
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector

    key = b"\xAA"*32
    FS  = 48000
    SECS = 5

    np.random.seed(52)
    noise = 0.01 * np.random.randn(FS * SECS).astype(np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(noise)

    print(f"Signal power: {np.mean(noise**2):.6f}")
    print(f"Watermark power: {np.mean((wm-noise)**2):.6f}")
    print(f"SNR: {10*np.log10(np.mean(noise**2)/np.mean((wm-noise)**2)):.1f} dB")

    rx = WatermarkDetector(key)
    ok = rx.verify(wm, FS)
    assert ok is True, "Quiet noise roundtrip still failing (see [LLR ALIGN] logs above)"
