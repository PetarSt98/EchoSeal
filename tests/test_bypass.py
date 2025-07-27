#!/usr/bin/env python3
"""
Bypass test that directly feeds perfect synthetic frames to the detector.
This isolates the LLR calculation and polar decoding from synchronization issues.
"""

import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc, decode as polar_dec


def test_perfect_synthetic_frame():
    """Test with a perfectly reconstructed frame that bypasses all sync issues."""

    print("=" * 60)
    print("PERFECT SYNTHETIC FRAME TEST")
    print("=" * 60)

    key = b"\xAA" * 32
    frame_ctr = 0

    # Create TX
    tx = WatermarkEmbedder(key)
    tx.frame_ctr = frame_ctr

    # Generate payload exactly as TX does
    payload_bytes = tx._build_payload()
    print(f"Payload: {payload_bytes[:8].hex()}")

    # Encode exactly as TX does
    data_bits = polar_enc(payload_bytes, N=1024, K=448)
    print(f"Data bits (first 16): {data_bits[:16]}")

    # Get PN sequence exactly as TX does
    sec = SecureChannel(key)
    frame_len = 63 + len(data_bits)
    pn_full = sec.pn_bits(frame_ctr, frame_len)
    pn_payload = pn_full[63:]
    print(f"PN payload (first 16): {pn_payload[:16]}")

    # Create symbols exactly as TX does (without filtering effects)
    preamble = np.array([1, 0, 1] * 21, dtype=np.uint8)[:63]
    preamble_symbols = (2 * preamble - 1).astype(np.float32)

    data_symbols = (2 * data_bits - 1).astype(np.float32)
    pn_symbols = (2 * pn_payload - 1).astype(np.float32)
    spread_symbols = data_symbols * pn_symbols

    # Perfect frame without any filtering
    perfect_frame = np.concatenate([preamble_symbols, spread_symbols])
    print(f"Perfect frame length: {len(perfect_frame)}")
    print(f"Perfect frame energy: {np.mean(perfect_frame ** 2):.6f}")

    # Test 1: Direct LLR calculation
    print("\n" + "-" * 40)
    print("TEST 1: Direct LLR calculation")
    print("-" * 40)

    payload_part = perfect_frame[63:]  # Extract payload
    despread = payload_part * pn_symbols  # Despread

    print(f"Despread symbols (first 16): {despread[:16]}")
    print(f"Original data symbols (first 16): {data_symbols[:16]}")
    print(f"Perfect match: {np.allclose(despread, data_symbols)}")

    # Calculate LLR directly
    llr = despread * 8.0  # Simple scaling
    llr = np.clip(llr, -30.0, 30.0)

    # Pad to 1024 if needed
    if len(llr) < 1024:
        llr_padded = np.zeros(1024, dtype=np.float32)
        llr_padded[:len(llr)] = llr
        llr = llr_padded

    print(f"LLR stats: mean={np.mean(llr):.3f}, std={np.std(llr):.3f}")

    # Test polar decoding
    decoded = polar_dec(llr)
    if decoded is not None:
        print(f"✓ Polar decoding SUCCESS")
        print(f"Payload match: {decoded == payload_bytes}")
        if decoded != payload_bytes:
            print(f"Expected: {payload_bytes[:16].hex()}")
            print(f"Got:      {decoded[:16].hex()}")
    else:
        print(f"✗ Polar decoding FAILED")

    # Test 2: Use detector's LLR function directly
    print("\n" + "-" * 40)
    print("TEST 2: Detector LLR function")
    print("-" * 40)

    detector = WatermarkDetector(key)
    detector_llr = detector._llr(perfect_frame, frame_ctr)

    print(f"Detector LLR stats: mean={np.mean(detector_llr):.3f}, std={np.std(detector_llr):.3f}")

    detector_decoded = polar_dec(detector_llr)
    if detector_decoded is not None:
        print(f"✓ Detector LLR SUCCESS")
        print(f"Payload match: {detector_decoded == payload_bytes}")
    else:
        print(f"✗ Detector LLR FAILED")

    # Test 3: Full detector with perfect frame
    print("\n" + "-" * 40)
    print("TEST 3: Full detector verify_raw_frame")
    print("-" * 40)

    result = detector.verify_raw_frame(perfect_frame)
    print(f"Full detector result: {result}")

    return result


def test_with_actual_tx_frame():
    """Test with actual TX-generated frame."""

    print("\n" + "=" * 60)
    print("ACTUAL TX FRAME TEST")
    print("=" * 60)

    key = b"\xAA" * 32

    # Generate actual TX frame
    tx = WatermarkEmbedder(key)
    tx_frame = tx._make_frame_chips()

    print(f"TX frame length: {len(tx_frame)}")
    print(f"TX frame energy: {np.mean(tx_frame ** 2):.6f}")

    # Test detection
    detector = WatermarkDetector(key)
    result = detector.verify_raw_frame(tx_frame)

    print(f"Detection result: {result}")
    return result


if __name__ == "__main__":
    print("Running bypass tests to isolate the problem...\n")

    test1_result = test_perfect_synthetic_frame()
    test2_result = test_with_actual_tx_frame()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Perfect synthetic frame: {'PASS' if test1_result else 'FAIL'}")
    print(f"Actual TX frame:         {'PASS' if test2_result else 'FAIL'}")

    if test1_result and not test2_result:
        print("\n→ Issue is in TX signal generation (filtering/scaling)")
    elif not test1_result and not test2_result:
        print("\n→ Issue is in detector LLR calculation or polar decoding")
    elif test1_result and test2_result:
        print("\n→ Both work! Issue might be in synchronization/peak detection")
    else:
        print("\n→ Inconsistent results - need more investigation")