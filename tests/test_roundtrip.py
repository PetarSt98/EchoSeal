
import numpy as np
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector, mseq_63
from rtwm.polar_fast import N_DEFAULT, K_DEFAULT
from rtwm.utils import lin_to_db, db_to_lin
from rtwm.fastpolar import PolarCode
from scipy.signal import chirp

FS, SECS = 48_000, 5

def test_decoder_manually():
    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8, crc_size=8)
    bits = np.random.randint(0, 2, K_DEFAULT - 8, dtype=np.uint8)
    enc = pc.encode(bits)
    llr = np.where(enc == 1, 10.0, -10.0).astype(np.float32)
    dec_bits, ok = pc.decode(llr)
    assert ok and np.all(bits == dec_bits[:bits.size])

def test_polar_loopback():
    from rtwm.polar_fast import encode, decode
    import os
    payload = os.urandom(55)  # must be exactly 56 bytes = 448 bits
    print(f"[TEST] Original payload[:8]: {payload[:8].hex()}")

    chips = encode(payload)
    llr = np.where(chips == 1, 10.0, -10.0).astype(np.float32)
    recovered = decode(llr)

    assert recovered == payload, f"Recovered: {recovered}, Original: {payload}"


def test_simple_watermark():
    """Simplified test to isolate the issue."""
    key = b"\xAA" * 32

    # Create a simple test signal - just noise
    np.random.seed(42)
    test_signal = 0.01 * np.random.randn(48000).astype(np.float32)  # 1 second of quiet noise

    # Embed watermark
    tx = WatermarkEmbedder(key)
    watermarked = tx.process(test_signal)

    # Check the watermark power
    watermark_only = watermarked - test_signal
    print(f"Test signal RMS: {np.sqrt(np.mean(test_signal ** 2)):.6f}")
    print(f"Watermark RMS: {np.sqrt(np.mean(watermark_only ** 2)):.6f}")
    print(f"Total RMS: {np.sqrt(np.mean(watermarked ** 2)):.6f}")

    # Try to detect
    detector = WatermarkDetector(key)

    # First, try with the pure watermark signal (no host signal)
    result_pure = detector.verify(watermark_only, 48000)
    print(f"Pure watermark detection: {result_pure}")

    # Then try with the mixed signal
    result_mixed = detector.verify(watermarked, 48000)
    print(f"Mixed signal detection: {result_mixed}")

    return result_pure or result_mixed


def test_polar_codec_only():
    """Test polar encoding/decoding in isolation."""
    from rtwm.polar_fast import encode as polar_enc, decode as polar_dec
    import numpy as np

    # Test with known data
    test_payload = b"A" * 55  # 55 bytes of 'A'
    print(f"Original payload: {test_payload[:8]}")

    # Encode
    bits = polar_enc(test_payload)
    print(f"Encoded bits[:8]: {bits[:8]}")
    print(f"Encoded bits[-8:]: {bits[-8:]}")

    # Perfect channel (no noise)
    llr = 2.0 * (2.0 * bits.astype(np.float32) - 1.0)  # Strong LLRs

    # Decode
    recovered = polar_dec(llr)
    print(f"Recovered payload: {recovered[:8] if recovered else None}")
    print(f"Polar codec test: {'PASS' if recovered == test_payload else 'FAIL'}")

    assert recovered == test_payload, "Polar codec test failed!"

# Also test with a much simpler signal
def test_minimal_signal():
    """Test with minimal complexity."""
    key = b"\xAA" * 32

    # Just zeros - pure watermark
    silence = np.zeros(96000, dtype=np.float32)  # 2 seconds

    tx = WatermarkEmbedder(key)
    watermarked = tx.process(silence)

    print(f"Watermarked silence RMS: {np.sqrt(np.mean(watermarked ** 2)):.6f}")

    detector = WatermarkDetector(key)
    result = detector.verify(watermarked, 48000)
    print(f"Silent watermark detection: {result}")

    return result


def test_tx_rx_end_to_end_minimal():
    """Minimal test that should work."""
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    from scipy.signal import chirp
    import numpy as np

    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)
    print(f"Signal power: {np.mean(speech ** 2):.6f}")
    print(f"Watermark power: {np.mean((wm - speech) ** 2):.6f}")
    print(f"SNR: {10 * np.log10(np.mean(speech ** 2) / np.mean((wm - speech) ** 2)):.1f} dB")
    if np.array_equal(wm, speech):
        print("WARNING: Watermark not embedded!")
    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    assert result is True


def test_tx_rx_silence_roundtrip():
    """Roundtrip test with silence to confirm the system works without host signal."""
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    import numpy as np

    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)

    # Use silence instead of chirp
    silence = np.zeros(FS * SECS, dtype=np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(silence)

    print(f"Signal power: {np.mean(silence ** 2):.6f}")
    print(f"Watermark power: {np.mean((wm - silence) ** 2):.6f}")
    print(f"SNR: {10 * np.log10(np.mean(silence ** 2 + 1e-12) / np.mean((wm - silence) ** 2)):.1f} dB")
    print(f"Watermark RMS: {np.sqrt(np.mean(wm ** 2)):.6f}")

    if np.array_equal(wm, silence):
        print("WARNING: Watermark not embedded!")

    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    assert result is True, "Silence roundtrip test failed!"
    print("SUCCESS: Silence roundtrip test passed!")


def test_tx_rx_very_quiet_noise_roundtrip():
    """Roundtrip test with very quiet background noise."""
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    import numpy as np

    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)

    # Very quiet white noise background
    noise = 0.01 * np.random.randn(FS * SECS).astype(np.float32)
    tx = WatermarkEmbedder(key)
    wm = tx.process(noise)

    print(f"Signal power: {np.mean(noise ** 2):.6f}")
    print(f"Watermark power: {np.mean((wm - noise) ** 2):.6f}")
    print(f"SNR: {10 * np.log10(np.mean(noise ** 2) / np.mean((wm - noise) ** 2)):.1f} dB")

    if np.array_equal(wm, noise):
        print("WARNING: Watermark not embedded!")

    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    assert result is True, "Quiet noise roundtrip test failed!"
    print("SUCCESS: Quiet noise roundtrip test passed!")


def test_tx_rx_low_frequency_tone_roundtrip():
    """Roundtrip test with low frequency tone that shouldn't interfere."""
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    import numpy as np

    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)

    # 1 kHz tone at moderate level - well below watermark bands (4kHz+)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(tone)

    print(f"Signal power: {np.mean(tone ** 2):.6f}")
    print(f"Watermark power: {np.mean((wm - tone) ** 2):.6f}")
    print(f"SNR: {10 * np.log10(np.mean(tone ** 2) / np.mean((wm - tone) ** 2)):.1f} dB")

    if np.array_equal(wm, tone):
        print("WARNING: Watermark not embedded!")

    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    assert result is True, "Low frequency tone roundtrip test failed!"
    print("SUCCESS: Low frequency tone roundtrip test passed!")

def test_tx_rx_end_to_end_debug():
    """Test with debug output."""
    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)

    print(f"Input signal: shape={speech.shape}, dtype={speech.dtype}")
    print(f"Input signal stats: mean={np.mean(speech):.4f}, std={np.std(speech):.4f}")

    tx = WatermarkEmbedder(key)

    # Check embedder state
    print(f"TX session nonce: {tx._session_nonce.hex()}")
    print(f"TX initial frame_ctr: {tx.frame_ctr}")

    wm = tx.process(speech)

    print(f"TX final frame_ctr: {tx.frame_ctr}")
    print(f"Watermarked signal: shape={wm.shape}, dtype={wm.dtype}")
    print(f"Watermarked stats: mean={np.mean(wm):.4f}, std={np.std(wm):.4f}")

    # Check if watermark was actually added
    diff = wm - speech
    print(f"Watermark component: mean={np.mean(diff):.6f}, std={np.std(diff):.6f}")

    rx = WatermarkDetector(key)

    # Add a method to check if preambles match
    tx_preamble = tx.mseq_63()
    rx_preamble = mseq_63()  # from detector.py
    print(f"Preambles match: {np.array_equal(tx_preamble, rx_preamble)}")
    if not np.array_equal(tx_preamble, rx_preamble):
        print(f"TX preamble[:10]: {tx_preamble[:10]}")
        print(f"RX preamble[:10]: {rx_preamble[:10]}")

    # Try detection
    result = rx.verify(wm, FS)
    print(f"Detection result: {result}")

    return result


def test_tx_rx_end_to_end_fixed():
    """Test complete transmit -> receive cycle."""
    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)

    # Create embedder and embed watermark
    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)

    # Create detector and verify
    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    # Add assertion with helpful error message
    assert result is True, f"Detection failed. TX generated {tx.frame_ctr} frames"

    return result


def test_minimal_polarcodel_roundtrip():
    from rtwm.polar_fast import encode, decode
    payload = bytes([i & 0xFF for i in range(55)])  # e.g., 0,1,2,...,54
    enc = encode(payload)
    print("Encoded (first 32):", enc[:32])

    # Simulate perfect channel, so LLR = large values with correct sign
    llr = np.where(enc == 0, -10.0, 10.0)  # Strong LLRs, noise-free

    decoded_bytes = decode(llr)
    print("Decoded:", decoded_bytes.hex() if decoded_bytes else None)
    print("Payload:", payload.hex())

    assert decoded_bytes == payload, "Polar roundtrip failed even without channel"

def test_different_signal_levels():
    """Test detection at different signal amplitudes."""
    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)

    # Test different amplitude levels
    amplitudes = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    for amp in amplitudes:
        print(f"\nTesting amplitude: {amp}")
        speech = amp * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)

        tx = WatermarkEmbedder(key)
        wm = tx.process(speech)

        # Check watermark strength
        watermark = wm - speech
        wm_std = np.std(watermark)
        print(f"  Watermark std: {wm_std:.6f}")

        result = WatermarkDetector(key).verify(wm, FS)
        print(f"  Detection: {'✓' if result else '✗'}")

def test_watermark_power_bounds():
    """Verify watermark power is within acceptable bounds."""
    key = b"\x33" * 32
    speech = (np.random.randn(FS * SECS) * 0.1).astype(np.float32)
    wm = WatermarkEmbedder(key).process(speech)

    # Calculate watermark-to-speech power ratio
    watermark_signal = wm - speech
    speech_power = np.mean(speech ** 2)
    watermark_power = np.mean(watermark_signal ** 2)

    ratio_db = lin_to_db(np.sqrt(watermark_power) / np.sqrt(speech_power))

    # Watermark should be significantly below speech level but detectable
    assert -40 < ratio_db < -10, f"Watermark power ratio {ratio_db:.1f} dB out of range"


def test_multiple_frames():
    """Test detection across multiple watermark frames."""
    key = b"\x44" * 32
    # Create longer audio to span multiple frames
    speech = (np.random.randn(FS * 5) * 0.05).astype(np.float32)
    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)

    # Should detect even with longer audio
    assert WatermarkDetector(key).verify(wm, FS) is True

    # Should also detect with subsections
    mid_section = wm[FS:FS * 4]  # 3 seconds from middle
    assert WatermarkDetector(key).verify(mid_section, FS) is True


def test_basic_roundtrip():
    """Test if basic roundtrip works after preamble fix."""
    import numpy as np
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector

    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    # Test with noise
    np.random.seed(42)
    signal = 0.1 * np.random.randn(FS * SECS).astype(np.float32)

    tx = WatermarkEmbedder(key)
    wm = tx.process(signal)

    rx = WatermarkDetector(key)
    result = rx.verify(wm, FS)

    assert result is True, "Basic roundtrip should work now!"
    print("✅ Basic roundtrip WORKS!")
    return True