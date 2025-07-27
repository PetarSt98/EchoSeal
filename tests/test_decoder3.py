#!/usr/bin/env python3
"""
Test the fixed detector implementation.
"""

import numpy as np
import sys
import os

# Add the module path if needed
# sys.path.insert(0, '/path/to/your/rtwm/module')

from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector

# Test parameters
KEY = b"\xAA" * 32
FS = 48_000


def test_perfect_frame():
    """Test detection on a perfect synthetic frame."""
    print("\n" + "=" * 60)
    print("TEST: Perfect Frame Detection")
    print("=" * 60)

    # Generate one perfect frame
    tx = WatermarkEmbedder(KEY)
    chips = tx._make_frame_chips()

    # Test detection with the raw frame
    detector = WatermarkDetector(KEY)
    result = detector.verify_raw_frame(chips)

    print(f"\nResult: {'PASS' if result else 'FAIL'}")
    return result


def test_with_padding():
    """Test detection when frame is embedded in silence."""
    print("\n" + "=" * 60)
    print("TEST: Frame with Padding")
    print("=" * 60)

    # Generate frame and embed in silence
    tx = WatermarkEmbedder(KEY)
    chips = tx._make_frame_chips()

    # Create 3-second silent buffer and insert frame at start
    signal_len = int(3.0 * FS)
    signal = np.zeros(signal_len, dtype=np.float32)
    signal[:len(chips)] = chips

    # Test detection
    detector = WatermarkDetector(KEY)
    result = detector.verify(signal, FS)

    print(f"\nResult: {'PASS' if result else 'FAIL'}")
    return result


def debug_signal_processing():
    """Debug the signal processing chain."""
    print("\n" + "=" * 60)
    print("DEBUG: Signal Processing Chain")
    print("=" * 60)

    from rtwm.crypto import SecureChannel
    from rtwm.polar_fast import encode as polar_enc

    # Replicate TX process
    tx = WatermarkEmbedder(KEY)
    sec = SecureChannel(KEY)

    # Generate payload and encode
    payload_bytes = tx._build_payload()
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype="u1"))
    data_bits = polar_enc(payload_bytes, N=1024, K=448)

    # Get PN sequence
    frame_len = 63 + len(data_bits)
    pn_full = sec.pn_bits(0, frame_len)
    pn_payload = pn_full[63:]

    # Check spreading operation
    spread_bits = data_bits ^ pn_payload  # XOR spreading
    symbols = 2 * spread_bits - 1  # Map to ±1

    print(f"Data bits (first 16): {data_bits[:16]}")
    print(f"PN payload (first 16): {pn_payload[:16]}")
    print(f"Spread bits (first 16): {spread_bits[:16]}")
    print(f"Symbols (first 16): {symbols[:16]}")

    # Test despreading
    recovered_spread = ((symbols + 1) // 2).astype(np.uint8)  # Map back to bits
    recovered_data = recovered_spread ^ pn_payload  # Despread

    print(f"Recovered spread (first 16): {recovered_spread[:16]}")
    print(f"Recovered data (first 16): {recovered_data[:16]}")
    print(f"Data recovery correct: {np.array_equal(data_bits, recovered_data)}")

    return np.array_equal(data_bits, recovered_data)


if __name__ == "__main__":
    print("Testing Fixed Detector Implementation")

    try:
        # Run tests
        debug_ok = debug_signal_processing()
        test1_ok = test_perfect_frame()
        test2_ok = test_with_padding()

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Signal processing debug: {'PASS' if debug_ok else 'FAIL'}")
        print(f"Perfect frame detection: {'PASS' if test1_ok else 'FAIL'}")
        print(f"Padded frame detection: {'PASS' if test2_ok else 'FAIL'}")

        if debug_ok and test1_ok and test2_ok:
            print("\n🎉 All tests passed! The detector fix works correctly.")
        else:
            print("\n❌ Some tests failed. The detector needs further debugging.")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()