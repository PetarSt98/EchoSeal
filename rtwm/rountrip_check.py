#!/usr/bin/env python3
"""
Fixed roundtrip test that ensures we're testing the actual TX/RX pipeline correctly.
"""

import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector


def fixed_roundtrip_test():
    """Test the TX/RX roundtrip with proper state management."""
    print("=" * 60)
    print("FIXED ROUNDTRIP TEST")
    print("=" * 60)

    key = b"\xAA" * 32

    # Test 1: Direct frame test
    print("\n1. DIRECT FRAME TEST")
    print("-" * 40)

    # Create embedder and generate ONE frame
    tx = WatermarkEmbedder(key)
    tx.frame_ctr = 0
    frame_chips = tx._make_frame_chips()

    print(f"Generated frame: length={len(frame_chips)}")
    print(f"Frame stats: mean={np.mean(frame_chips):.6f}, std={np.std(frame_chips):.6f}")

    # Create detector and test on the SAME frame
    rx = WatermarkDetector(key)

    # The detector expects a signal that might contain multiple frames
    # So let's create a signal with just our frame
    test_signal = np.zeros(len(frame_chips) * 3, dtype=np.float32)
    test_signal[len(frame_chips):2 * len(frame_chips)] = frame_chips

    result = rx.verify(test_signal, tx.p.fs)
    print(f"Detection result on padded signal: {result}")

    # Test 2: Using process() pipeline
    print("\n2. PROCESS() PIPELINE TEST")
    print("-" * 40)

    # Create fresh embedder
    tx2 = WatermarkEmbedder(key)

    # Create a carrier signal long enough for one frame
    carrier = np.zeros(1087 * 2, dtype=np.float32)  # 2 frames worth

    # Process should embed the watermark
    watermarked = tx2.process(carrier)

    print(f"Watermarked signal: length={len(watermarked)}")
    print(f"Signal power: {np.mean(watermarked ** 2):.6e}")

    # Create fresh detector
    rx2 = WatermarkDetector(key)
    result2 = rx2.verify(watermarked, tx2.p.fs)
    print(f"Detection result: {result2}")

    # Test 3: Minimal test - just one frame, no padding
    print("\n3. MINIMAL FRAME TEST")
    print("-" * 40)

    # Generate a frame with known counter
    tx3 = WatermarkEmbedder(key)
    tx3.frame_ctr = 0

    # Get the band for frame 0
    from rtwm.utils import choose_band
    band = choose_band(key, 0)
    print(f"Using band: {band}")

    # Generate frame
    frame = tx3._make_frame_chips()

    # Try raw frame verification
    result3 = rx.verify_raw_frame(frame)
    print(f"Raw frame verification: {result3}")

    # Debug: Let's check what's in the frame
    print(f"\nFrame analysis:")
    print(f"  First 10 samples: {frame[:10]}")
    print(f"  Preamble region std: {np.std(frame[:63]):.3f}")
    print(f"  Payload region std: {np.std(frame[63:]):.3f}")

    return result or result2 or result3


def test_simple_encode_decode():
    """Test just the encode/decode without filtering."""
    print("\n" + "=" * 60)
    print("SIMPLE ENCODE/DECODE TEST")
    print("=" * 60)

    key = b"\xAA" * 32

    # Create TX and generate payload
    from rtwm.embedder import WatermarkEmbedder
    from rtwm.crypto import SecureChannel
    from rtwm.polar_fast import encode as polar_enc, decode as polar_dec

    tx = WatermarkEmbedder(key)
    tx.frame_ctr = 0

    # Generate payload
    payload = tx._build_payload()
    print(f"Payload: {len(payload)} bytes")

    # Encode
    data_bits = polar_enc(payload, N=1024, K=448)
    print(f"Encoded bits: {len(data_bits)}")

    # Convert to symbols
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0

    # Add some noise
    noise = np.random.normal(0, 0.1, size=len(data_symbols))
    received = data_symbols + noise

    # Convert back to LLR
    llr = received * 8.0  # Simple scaling

    # Decode
    decoded = polar_dec(llr)

    if decoded is not None:
        print(f"Decoded: {len(decoded)} bytes")
        print(f"Match: {decoded == payload}")

        # Try to decrypt
        sec = SecureChannel(key)
        try:
            plain = sec.open(decoded)
            print(f"Decrypted successfully!")
            print(f"Starts with ESAL: {plain.startswith(b'ESAL')}")
            ctr = int.from_bytes(plain[4:8], "big")
            print(f"Counter: {ctr}")
        except Exception as e:
            print(f"Decrypt failed: {e}")
    else:
        print("Polar decode failed!")


if __name__ == "__main__":
    success = fixed_roundtrip_test()
    print(f"\nOverall success: {success}")

    # Also run the simple encode/decode test
    test_simple_encode_decode()