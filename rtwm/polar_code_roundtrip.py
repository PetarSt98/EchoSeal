#!/usr/bin/env python3
"""
Adapted diagnostic test for fastpolar_old.py implementation.
"""

import numpy as np
import secrets
from scipy.signal import correlate, lfilter


def test_polar_code_alone():
    """Test polar encoding/decoding in isolation."""
    print("=" * 60)
    print("TEST 1: POLAR CODE IN ISOLATION")
    print("=" * 60)

    from rtwm.fastpolar import PolarCode

    # Create polar code instance
    N = 1024
    K = 448  # includes CRC
    pc = PolarCode(N=N, K=K, list_size=8, crc_size=8)

    # Create a known payload (55 bytes = 440 bits, which becomes 448 with CRC)
    payload_bytes = b"A" * 55
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    info_bits = payload_bits[:440]  # 440 info bits (K - crc_size)

    print(f"Original payload: {payload_bytes.hex()[:32]}...")
    print(f"Info bits length: {len(info_bits)}")

    # Encode
    encoded = pc.encode(info_bits)
    print(f"Encoded length: {len(encoded)} bits")
    print(f"Encoded (first 32): {encoded[:32]}")

    # Perfect channel (no noise)
    llr = np.where(encoded == 0, -10.0, 10.0)

    # Decode
    decoded_bits, ok = pc.decode(llr)

    if ok and np.array_equal(decoded_bits, info_bits):
        print("✓ Polar code roundtrip PASSED")
        return True
    else:
        print("✗ Polar code roundtrip FAILED")
        if ok:
            print(f"CRC passed but bits don't match")
        else:
            print(f"CRC failed")
        return False


def test_polar_with_noise():
    """Test polar code with some noise."""
    print("\n" + "=" * 60)
    print("TEST 1b: POLAR CODE WITH NOISE")
    print("=" * 60)

    from rtwm.fastpolar import PolarCode

    # Create polar code instance with list decoding
    N = 1024
    K = 448
    pc = PolarCode(N=N, K=K, list_size=8, crc_size=8)

    # Create payload
    payload_bytes = b"B" * 55
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    info_bits = payload_bits[:440]

    # Encode
    encoded = pc.encode(info_bits)

    # Add some noise
    llr = np.where(encoded == 0, -5.0, 5.0)
    noise = np.random.normal(0, 1.0, size=len(llr))
    llr_noisy = llr + noise

    # Decode with list decoding
    decoded_bits, ok = pc.decode(llr_noisy)

    if ok and np.array_equal(decoded_bits, info_bits):
        print("✓ Polar code with noise PASSED (list decoding worked)")
        return True
    elif ok:
        print("⚠ CRC passed but bits don't match exactly")
        bit_errors = np.sum(decoded_bits != info_bits)
        print(f"  Bit errors: {bit_errors}/440")
        return False
    else:
        print("✗ Polar code with noise FAILED (CRC failed)")
        return False


def test_crypto_channel():
    """Test the crypto seal/open operations."""
    print("\n" + "=" * 60)
    print("TEST 2: CRYPTO CHANNEL")
    print("=" * 60)

    from rtwm.crypto import SecureChannel

    key = b"\xAA" * 32
    sec = SecureChannel(key)

    # Test message
    plaintext = b"ESAL" + (0).to_bytes(4, "big") + secrets.token_bytes(19)
    print(f"Plaintext ({len(plaintext)} bytes): {plaintext.hex()}")

    # Seal
    ciphertext = sec.seal(plaintext)
    print(f"Ciphertext ({len(ciphertext)} bytes): {ciphertext.hex()[:32]}...")

    # Open
    try:
        decrypted = sec.open(ciphertext)
        if decrypted == plaintext:
            print("✓ Crypto roundtrip PASSED")
            return True
        else:
            print("✗ Crypto roundtrip FAILED - wrong plaintext")
            return False
    except Exception as e:
        print(f"✗ Crypto roundtrip FAILED - {e}")
        return False


def test_pn_sequence_generation():
    """Test PN sequence generation consistency."""
    print("\n" + "=" * 60)
    print("TEST 3: PN SEQUENCE GENERATION")
    print("=" * 60)

    from rtwm.crypto import SecureChannel

    key = b"\xAA" * 32
    sec = SecureChannel(key)

    # Generate PN for frame 0
    pn1 = sec.pn_bits(0, 1087)
    pn2 = sec.pn_bits(0, 1087)

    if np.array_equal(pn1, pn2):
        print("✓ PN sequence is deterministic")
    else:
        print("✗ PN sequence is NOT deterministic!")
        return False

    # Check PN properties
    print(f"PN mean: {np.mean(pn1):.3f} (should be ~0.5)")
    print(f"PN std: {np.std(pn1):.3f}")

    return True


def test_frame_generation_and_detection():
    """Test single frame generation and detection."""
    print("\n" + "=" * 60)
    print("TEST 4: SINGLE FRAME GENERATION AND DETECTION")
    print("=" * 60)

    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    from rtwm.utils import choose_band, butter_bandpass

    key = b"\xAA" * 32

    # Generate a single frame
    tx = WatermarkEmbedder(key)
    tx.frame_ctr = 0  # Force frame 0

    # Store the session nonce
    tx._session_nonce = b"12345678"  # Fixed 8-byte nonce

    # Generate frame
    frame = tx._make_frame_chips()
    print(f"Frame length: {len(frame)}")
    print(f"Frame std: {np.std(frame):.6f}")

    # Try to detect it directly
    rx = WatermarkDetector(key)

    # Get the band for frame 0
    band = choose_band(key, 0)
    print(f"Frame 0 band: {band}")

    # Apply the same filter the detector would use
    b, a = butter_bandpass(*band, 48000, order=4)

    # The frame is already filtered by the embedder, but let's check
    # if we can find the preamble
    PREAMBLE = np.array([1, 0, 1] * 21, dtype=np.uint8)[:63]
    preamble_symbols = 2.0 * PREAMBLE.astype(np.float32) - 1.0

    # Create filtered preamble template
    dummy_frame = np.zeros(1087, dtype=np.float32)
    dummy_frame[:63] = preamble_symbols
    dummy_filtered = lfilter(b, a, dummy_frame)
    preamble_template = dummy_filtered[:63].copy()
    preamble_template /= np.sqrt(np.sum(preamble_template ** 2) + 1e-12)

    # Correlate
    corr = correlate(frame, preamble_template, mode='valid')
    max_corr = np.max(np.abs(corr))
    max_idx = np.argmax(np.abs(corr))

    print(f"Max correlation: {max_corr:.3f} at index {max_idx}")

    if max_corr > 0.5 and max_idx == 0:
        print("✓ Preamble detected at correct position")
    else:
        print("✗ Preamble detection issue")
        return False

    # Now try to decode the frame with the correct counter
    result = rx._try_decode_frame(frame, 0)

    if result:
        print("✓ Frame decoded successfully")
        return True
    else:
        print("✗ Frame decoding failed")

        # Let's debug the LLR calculation
        print("\nDEBUGGING LLR CALCULATION:")

        # Get PN sequence
        from rtwm.crypto import SecureChannel
        sec = SecureChannel(key)
        pn_full = sec.pn_bits(0, 1087)
        pn_payload = pn_full[63:]  # Skip preamble

        # Extract payload
        payload_received = frame[63:].copy()

        # Convert PN to symbols
        pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

        # Despread
        despread = payload_received * pn_symbols

        print(f"Despread mean: {np.mean(despread):.6f}")
        print(f"Despread std: {np.std(despread):.6f}")

        # Calculate simple LLRs
        llr = despread * 4.0  # Simple scaling
        print(f"LLR range: [{np.min(llr):.2f}, {np.max(llr):.2f}]")
        print(f"LLR std: {np.std(llr):.2f}")

        return False


def test_full_pipeline_minimal():
    """Test the absolute minimal case."""
    print("\n" + "=" * 60)
    print("TEST 5: MINIMAL FULL PIPELINE")
    print("=" * 60)

    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector

    key = b"\xAA" * 32

    # Create embedder with fixed session nonce
    tx = WatermarkEmbedder(key)
    tx._session_nonce = b"12345678"  # Override with fixed nonce

    # Generate watermark in silence
    silence = np.zeros(48000 * 2, dtype=np.float32)  # 2 seconds
    watermarked = tx.process(silence)

    print(f"Generated {tx.frame_ctr} frames")
    print(f"Watermark power: {np.mean(watermarked ** 2):.6e}")

    # Try to detect
    rx = WatermarkDetector(key)
    result = rx.verify(watermarked, 48000)

    if result:
        print("✓ Full pipeline PASSED")
        return True
    else:
        print("✗ Full pipeline FAILED")
        return False


def run_all_tests():
    """Run all diagnostic tests."""
    results = []

    results.append(("Polar Code", test_polar_code_alone()))
    results.append(("Polar Code (Noise)", test_polar_with_noise()))
    results.append(("Crypto Channel", test_crypto_channel()))
    results.append(("PN Sequence", test_pn_sequence_generation()))
    results.append(("Frame Gen/Detect", test_frame_generation_and_detection()))
    results.append(("Full Pipeline", test_full_pipeline_minimal()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")

    all_passed = all(r[1] for r in results)

    if not all_passed:
        print("\nDIAGNOSIS:")
        failed_tests = [name for name, passed in results if not passed]
        for test_name in failed_tests:
            if test_name == "Polar Code":
                print("- Basic polar encoding/decoding has issues")
            elif test_name == "Polar Code (Noise)":
                print("- Polar code list decoding not handling noise well")
            elif test_name == "Crypto Channel":
                print("- Crypto seal/open has issues")
            elif test_name == "PN Sequence":
                print("- PN sequence generation not deterministic")
            elif test_name == "Frame Gen/Detect":
                print("- Frame generation/detection mismatch")
            elif test_name == "Full Pipeline":
                print("- Multi-frame or integration issues")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()