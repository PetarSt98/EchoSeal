#!/usr/bin/env python3

"""
Test script to verify the detector fixes.
Run this after applying the detector fixes.
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


def test_basic_detection():
    """Basic test: can detector find a clean embedded frame?"""
    print("\n" + "=" * 50)
    print("TEST 1: Basic Detection")
    print("=" * 50)

    # Generate a clean frame
    tx = WatermarkEmbedder(KEY)
    chips = tx._make_frame_chips()

    # Create a longer signal with the frame at the start
    signal_len = int(3.0 * FS)  # 3 seconds
    signal = np.zeros(signal_len, dtype=np.float32)
    signal[:len(chips)] = chips

    # Test detection
    detector = WatermarkDetector(KEY)
    result = detector.verify(signal, FS)

    print(f"Detection result: {result}")
    return result


def test_signal_analysis():
    """Analyze the signal to understand what's happening."""
    print("\n" + "=" * 50)
    print("TEST 2: Signal Analysis")
    print("=" * 50)

    # Run the debug function
    try:
        from tx_rx_comparison import debug_tx_rx_mismatch
        results = debug_tx_rx_mismatch(KEY)

        print(f"\nSummary:")
        print(f"- Correlation between despread and original data: {results['correlation']:.6f}")

        if abs(results['correlation']) > 0.5:
            print("✅ Signal processing looks reasonable")
        else:
            print("❌ Signal processing has issues")

    except ImportError:
        print("Debug function not available, skipping detailed analysis")


def test_llr_scaling():
    """Test different LLR scaling approaches."""
    print("\n" + "=" * 50)
    print("TEST 3: LLR Scaling")
    print("=" * 50)

    # This would require modifying the detector to return LLR values
    # For now, just run basic detection and check debug output

    tx = WatermarkEmbedder(KEY)
    chips = tx._make_frame_chips()

    signal_len = int(1.2 * len(chips))  # Just a bit longer than one frame
    signal = np.zeros(signal_len, dtype=np.float32)
    signal[:len(chips)] = chips

    detector = WatermarkDetector(KEY)

    # Enable detailed logging by calling verify_raw_frame for clean signal
    result = detector.verify_raw_frame(signal[:len(chips)])
    print(f"Raw frame detection: {result}")

    return result


if __name__ == "__main__":
    print("Testing Fixed Detector")
    print("Current working directory:", os.getcwd())

    try:
        # Run tests
        test1_result = test_basic_detection()
        test_signal_analysis()
        test3_result = test_llr_scaling()

        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(f"Basic detection: {'PASS' if test1_result else 'FAIL'}")
        print(f"Raw frame detection: {'PASS' if test3_result else 'FAIL'}")

        if test1_result and test3_result:
            print("🎉 All tests passed! Detector is working.")
        else:
            print("❌ Some tests failed. Check the debug output above.")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()