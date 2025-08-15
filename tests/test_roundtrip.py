
import numpy as np
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector
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

def test_debug():
    from rtwm.crypto import SecureChannel
    sec = SecureChannel(b"\xAA" * 32)

    # PN that the **receiver** will use for the first frame
    rx_pn = sec.pn_bits(0, 1087)[63:]  # skip the pre-amble

    from rtwm.embedder import WatermarkEmbedder
    tx = WatermarkEmbedder(b"\xAA" * 32)

    # The sign TX actually multiplies the first payload chip with:
    first_sym_sign = (2 * tx._make_frame_chips().copy()[:64] > 0).astype(int)[63]

    print(rx_pn[0], first_sym_sign)

def test_tx_rx_end_to_end():
    """Test complete transmit -> receive cycle."""
    key = b"\xAA" * 32
    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)
    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)
    assert WatermarkDetector(key).verify(wm, FS) is True

def test_tx_rx_end_to_end_fixed():
    """Test complete transmit -> receive cycle with proper signal levels."""
    key = b"\xAA" * 32
    FS = 48000
    SECS = 5

    np.random.seed(52)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)

    # Use a more realistic signal amplitude (0.1 to 0.5 is typical for audio)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)

    print(f"Original signal stats:")
    print(f"  Mean: {np.mean(speech):.6f}")
    print(f"  Std: {np.std(speech):.6f}")
    print(f"  Max: {np.max(np.abs(speech)):.6f}")

    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)

    print(f"\nWatermarked signal stats:")
    print(f"  Mean: {np.mean(wm):.6f}")
    print(f"  Std: {np.std(wm):.6f}")
    print(f"  Max: {np.max(np.abs(wm)):.6f}")

    # Calculate SNR
    watermark_power = np.mean((wm - speech) ** 2)
    signal_power = np.mean(speech ** 2)
    snr_db = 10 * np.log10(signal_power / watermark_power)
    print(f"  SNR: {snr_db:.1f} dB (should be close to 10 dB)")

    result = WatermarkDetector(key).verify(wm, FS)
    print(f"\nDetection result: {result}")

    assert result is True

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
