#!/usr/bin/env python3
"""
Comprehensive verification test for the fixed TX/RX implementation.
This test verifies that the signal processing chain works correctly.
"""

import numpy as np
import sys
import os

# Import your modules (adjust paths as needed)
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc, decode as polar_dec
from rtwm.utils import choose_band, butter_bandpass
from scipy.signal import lfilter


def test_signal_processing_chain():
    """Test the complete signal processing chain step by step."""
    print("=" * 80)
    print("SIGNAL PROCESSING CHAIN VERIFICATION")
    print("=" * 80)

    key = b"\xAA" * 32
    frame_ctr = 0
    fs = 48000

    # Step 1: Generate payload and encode
    print("\n1. PAYLOAD AND ENCODING")
    print("-" * 40)

    tx = WatermarkEmbedder(key)
    tx.frame_ctr = frame_ctr
    payload_bytes = tx._build_payload()

    sec = SecureChannel(key)
    data_bits = polar_enc(payload_bytes, N=1024, K=448)

    print(f"Payload: {len(payload_bytes)} bytes")
    print(f"Data bits: {len(data_bits)} bits")
    print(f"Data bits (first 16): {data_bits[:16]}")

    # Step 2: Generate PN sequence
    print("\n2. PN SEQUENCE GENERATION")
    print("-" * 40)

    frame_len = 63 + len(data_bits)
    pn_full = sec.pn_bits(frame_ctr, frame_len)
    pn_payload = pn_full[63:]

    print(f"PN sequence length: {len(pn_full)} bits")
    print(f"PN payload length: {len(pn_payload)} bits")
    print(f"PN payload (first 16): {pn_payload[:16]}")

    # Step 3: Manual TX process (should match embedder)
    print("\n3. MANUAL TX PROCESS")
    print("-" * 40)

    # Symbol mapping: {0,1} → {-1,+1}
    preamble = np.array([1, 0, 1] * 21, dtype=np.uint8)[:63]
    preamble_symbols = 2.0 * preamble.astype(np.float32) - 1.0
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0
    pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

    # Spreading
    spread_payload = data_symbols * pn_symbols
    all_symbols = np.concatenate([preamble_symbols, spread_payload])

    # Bandpass filtering
    band = choose_band(key, frame_ctr)
    b, a = butter_bandpass(*band, fs)
    filtered_chips = lfilter(b, a, all_symbols)

    # Normalization
    energy = np.mean(filtered_chips ** 2)
    if energy > 1e-12:
        filtered_chips /= np.sqrt(energy)

    print(f"Band: {band}")
    print(f"Symbols length: {len(all_symbols)}")
    print(f"Data symbols (first 8): {data_symbols[:8]}")
    print(f"PN symbols (first 8): {pn_symbols[:8]}")
    print(f"Spread symbols (first 8): {spread_payload[:8]}")
    print(f"Filtered chips energy: {np.mean(filtered_chips ** 2):.6f}")

    # Step 4: Compare with actual TX output
    print("\n4. TX OUTPUT COMPARISON")
    print("-" * 40)

    tx_chips = tx._make_frame_chips()

    print(f"TX chips length: {len(tx_chips)}")
    print(f"Manual chips length: {len(filtered_chips)}")
    print(f"Shapes match: {tx_chips.shape == filtered_chips.shape}")

    if tx_chips.shape == filtered_chips.shape:
        max_diff = np.max(np.abs(tx_chips - filtered_chips))
        mean_diff = np.mean(np.abs(tx_chips - filtered_chips))
        print(f"Max difference: {max_diff:.8f}")
        print(f"Mean difference: {mean_diff:.8f}")
        print(f"Close match (atol=1e-6): {np.allclose(tx_chips, filtered_chips, atol=1e-6)}")

        if not np.allclose(tx_chips, filtered_chips, atol=1e-6):
            print("⚠️  TX implementation doesn't match manual process!")
            return False
        else:
            print("✅ TX implementation matches manual process!")

    # Step 5: RX despreading test
    print("\n5. RX DESPREADING TEST")
    print("-" * 40)

    # Extract payload from TX signal
    rx_payload = tx_chips[63:]  # Skip preamble

    # Despread with PN sequence
    min_len = min(len(rx_payload), len(pn_symbols))
    rx_payload_trimmed = rx_payload[:min_len]
    pn_symbols_trimmed = pn_symbols[:min_len]

    despread = rx_payload_trimmed * pn_symbols_trimmed

    print(f"RX payload length: {len(rx_payload)}")
    print(f"Despread length: {len(despread)}")
    print(f"Despread (first 8): {despread[:8]}")
    print(f"Original data symbols (first 8): {data_symbols[:8]}")

    # Check correlation
    correlation = np.corrcoef(despread, data_symbols[:len(despread)])[0, 1]
    print(f"Correlation with original data: {correlation:.6f}")

    if abs(correlation) < 0.8:
        print("⚠️  Low correlation - despreading may be incorrect!")
        return False
    else:
        print("✅ Good correlation - despreading working correctly!")

    # Step 6: LLR and Polar decoding test
    print("\n6. LLR AND POLAR DECODING TEST")
    print("-" * 40)

    # Simple LLR calculation
    llr_scale = 8.0
    llr = despread * llr_scale
    llr = np.clip(llr, -30.0, 30.0)

    # Pad to correct length for polar decoder
    if len(llr) < 1024:
        llr_padded = np.zeros(1024, dtype=np.float32)
        llr_padded[:len(llr)] = llr
        llr = llr_padded

    print(f"LLR stats: mean={np.mean(llr):.3f}, std={np.std(llr):.3f}")
    print(f"LLR range: [{llr.min():.3f}, {llr.max():.3f}]")

    # Try polar decoding
    decoded_bytes = polar_dec(llr)

    if decoded_bytes is not None:
        print("✅ Polar decoding: SUCCESS!")
        print(f"Decoded length: {len(decoded_bytes)} bytes")
        print(f"Original payload: {payload_bytes[:8].hex()}")
        print(f"Decoded payload:  {decoded_bytes[:8].hex()}")

        if payload_bytes == decoded_bytes:
            print("✅ Perfect payload match!")
            return True
        else:
            print("⚠️  Payload mismatch!")
            return False
    else:
        print("❌ Polar decoding failed!")

        # Try different scales
        print("Trying different LLR scales...")
        for scale in [1.0, 2.0, 4.0, 16.0, 32.0]:
            test_llr = np.clip(despread * scale, -30.0, 30.0)
            if len(test_llr) < 1024:
                test_llr_padded = np.zeros(1024, dtype=np.float32)
                test_llr_padded[:len(test_llr)] = test_llr
                test_llr = test_llr_padded

            test_decoded = polar_dec(test_llr)
            if test_decoded is not None and test_decoded == payload_bytes:
                print(f"✅ SUCCESS with scale {scale}!")
                return True
            elif test_decoded is not None:
                print(f"⚠️  Scale {scale}: decoded but wrong payload")
            else:
                print(f"❌ Scale {scale}: failed")

        return False


def test_full_tx_rx_roundtrip():
    """Test the complete TX/RX roundtrip."""
    print("\n" + "=" * 80)
    print("FULL TX/RX ROUNDTRIP TEST")
    print("=" * 80)

    key = b"\xAA" * 32

    # Generate TX signal
    tx = WatermarkEmbedder(key)
    tx.frame_ctr = 0
    chips = tx._make_frame_chips()

    # Create a longer signal
    signal_len = max(3000, len(chips) * 2)  # At least 2 frame lengths
    signal = np.zeros(signal_len, dtype=np.float32)
    signal[:len(chips)] = chips

    print(f"Generated signal: {len(signal)} samples")
    print(f"Frame length: {len(chips)} samples")

    # Test RX detection
    detector = WatermarkDetector(key)
    result = detector.verify(signal, 48000)

    print(f"Detection result: {'SUCCESS' if result else 'FAILED'}")

    return result


def main():
    """Run all verification tests."""
    print("TX/RX VERIFICATION TESTS")
    print("=" * 80)

    # Test 1: Signal processing chain
    test1_passed = test_signal_processing_chain()

    # Test 2: Full roundtrip
    test2_passed = test_full_tx_rx_roundtrip()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Signal processing chain: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Full TX/RX roundtrip:    {'PASS' if test2_passed else 'FAIL'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The TX/RX chain is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)