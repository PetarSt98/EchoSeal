#!/usr/bin/env python3
"""
Fixed debugger that ensures we compare the same watermark frame.
"""

import numpy as np
from scipy.signal import lfilter
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector, PRE_BITS, PRE_L, FRAME_LEN
from rtwm.crypto import SecureChannel
from rtwm.utils import choose_band, butter_bandpass
from rtwm.polar_fast import encode as polar_enc, decode as polar_dec


def fixed_debug_analysis(key=b"\xAA" * 32):
    """Debug TX/RX with consistent frame generation."""
    print("=" * 80)
    print("FIXED TX/RX DEBUG ANALYSIS")
    print("=" * 80)

    # Create embedder
    tx = WatermarkEmbedder(key)
    tx.frame_ctr = 0

    # Generate ONE frame and analyze it
    print("\n1. GENERATE SINGLE FRAME")
    print("-" * 40)

    # Store the current random state
    import secrets
    import random

    # Generate the actual frame
    actual_frame = tx._make_frame_chips()
    print(f"Generated frame: length={len(actual_frame)}")
    print(f"Frame stats: mean={np.mean(actual_frame):.6f}, std={np.std(actual_frame):.6f}")

    # Now let's manually reconstruct what SHOULD have been generated
    # We need to use the same payload that was used
    print("\n2. ANALYZE FRAME GENERATION")
    print("-" * 40)

    # We can't access the exact payload that was used, but we can test
    # with a fresh frame and ensure consistency
    tx2 = WatermarkEmbedder(key)
    tx2.frame_ctr = 0

    # Generate a frame and immediately test it
    test_frame = tx2._make_frame_chips()

    # Create detector
    rx = WatermarkDetector(key)

    # Test detection on this frame
    print("\n3. DETECTION TEST")
    print("-" * 40)

    # Method 1: Raw frame test
    result1 = rx.verify_raw_frame(test_frame)
    print(f"Raw frame verification: {result1}")

    # Method 2: Embedded in larger signal
    test_signal = np.zeros(len(test_frame) * 3, dtype=np.float32)
    test_signal[len(test_frame):2 * len(test_frame)] = test_frame

    result2 = rx.verify(test_signal, tx2.p.fs)
    print(f"Padded signal verification: {result2}")

    # Detailed analysis if both fail
    if not result1 and not result2:
        print("\n4. DETAILED FAILURE ANALYSIS")
        print("-" * 40)

        # Use the same band selection method as the system (derived band key)
        sec = SecureChannel(key)
        band_key = getattr(sec, "band_key", key)
        band = choose_band(band_key, 0)
        print(f"Band: {band}")

        # Check preamble correlation
        b, a = butter_bandpass(*band, tx2.p.fs, order=4)
        preamble_symbols = 2.0 * PRE_BITS.astype(np.float32) - 1.0
        preamble_filtered = lfilter(b, a, preamble_symbols)

        # Check correlation with frame start
        frame_preamble = test_frame[:PRE_L]
        denom = np.sqrt(np.sum(frame_preamble ** 2) * np.sum(preamble_filtered ** 2)) + 1e-12
        corr = float(np.dot(frame_preamble, preamble_filtered) / denom)
        print(f"Preamble correlation: {corr:.3f}")

        # Try manual LLR calculation
        pn_full = sec.pn_bits(0, FRAME_LEN)
        pn_payload = pn_full[PRE_L:]
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        payload_received = test_frame[PRE_L:]
        despread = payload_received * pn_symbols

        print(f"Despread stats: mean={np.mean(despread):.3f}, std={np.std(despread):.3f}")

        # Simple LLR
        llr = np.clip(despread * 8.0, -30.0, 30.0).astype(np.float32, copy=False)

        decoded = polar_dec(llr[:1024])
        if decoded is not None:
            try:
                plain = sec.open(decoded)
                print(f"Manual decode: SUCCESS - {plain[:8].hex()}")
            except Exception:
                print("Manual decode: Polar succeeded but decrypt failed")
        else:
            print("Manual decode: Polar decode failed")

    return result1 or result2


def test_filterless_roundtrip(key=b"\xAA" * 32):
    """Test without any filtering to isolate the issue."""
    print("\n" + "=" * 80)
    print("FILTERLESS ROUNDTRIP TEST")
    print("=" * 80)

    from rtwm.crypto import SecureChannel
    from rtwm.polar_fast import encode as polar_enc, decode as polar_dec

    sec = SecureChannel(key)

    # Generate payload
    frame_ctr = 0
    payload_data = b"ESAL" + frame_ctr.to_bytes(4, "big") + b"A" * 19
    payload = sec.seal(payload_data)
    print(f"Payload: {len(payload)} bytes")

    # Encode
    data_bits = polar_enc(payload, N=1024, K=448)

    # Generate PN
    pn_bits = sec.pn_bits(frame_ctr, 1024)

    # Spread
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0
    pn_symbols = 2.0 * pn_bits.astype(np.float32) - 1.0
    spread = data_symbols * pn_symbols

    # Add minimal noise
    received = spread + np.random.normal(0, 0.01, size=len(spread))

    # Despread
    despread = received * pn_symbols

    # Decode
    llr = (despread * 10.0).astype(np.float32, copy=False)
    decoded = polar_dec(llr)

    if decoded is not None and decoded == payload:
        print("✅ Filterless roundtrip: SUCCESS")
        return True
    else:
        print("❌ Filterless roundtrip: FAILED")
        return False


if __name__ == "__main__":
    # Run the fixed analysis
    success1 = fixed_debug_analysis()
    print(f"\nFixed analysis success: {success1}")

    # Run filterless test
    success2 = test_filterless_roundtrip()
    print(f"Filterless test success: {success2}")

    print(f"\nOverall success: {success1 or success2}")