
import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
from rtwm.polar_fast import N_DEFAULT, K_DEFAULT
from rtwm.utils import lin_to_db
from rtwm.fastpolar import PolarCode
from scipy.signal import chirp

FS, SECS = 48_000, 5

# def test_tx_rx_end_to_end():
#     key = b"\xAA" * 32
#     speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
#     tx = WatermarkEmbedder(key)
#     wm = tx.process(speech)
#     assert WatermarkDetector(key).verify(wm, FS) is True
#
# def test_watermark_power_bounds():
#     key = b"\x33"*32
#     speech = (np.random.randn(FS*SECS) * 0.1).astype(np.float32)
#     wm = WatermarkEmbedder(key).process(speech)
#     ratio_db = lin_to_db(np.sqrt(np.mean((wm-speech)**2)) /
#                          np.sqrt(np.mean(speech**2)))
#     assert -40 < ratio_db < -10

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


def test_tx_rx_end_to_end():
    """Test complete transmit -> receive cycle."""
    key = b"\xAA" * 32
    np.random.seed(42)
    t = np.linspace(0, SECS, FS * SECS, endpoint=False)
    speech = 0.1 * chirp(t, f0=300, f1=3500, t1=SECS, method='linear').astype(np.float32)
    tx = WatermarkEmbedder(key)
    wm = tx.process(speech)
    assert WatermarkDetector(key).verify(wm, FS) is True


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
