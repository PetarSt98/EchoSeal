#!/usr/bin/env python3
"""
Comprehensive debugging tool to identify TX/RX mismatches.
This will help us find exactly where the signal processing goes wrong.
"""

import numpy as np
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc, decode as polar_dec
from rtwm.utils import butter_bandpass
from scipy.signal import lfilter


def comprehensive_debug():
    """Debug every step of the TX/RX chain to find the mismatch."""

    print("=" * 80)
    print("COMPREHENSIVE TX/RX DEBUG ANALYSIS")
    print("=" * 80)

    key = b"\xAA" * 32
    frame_ctr = 0
    fs = 48000

    # Step 1: Generate identical payload in TX and RX
    print("\n1. PAYLOAD GENERATION")
    print("-" * 40)

    tx = WatermarkEmbedder(key)
    tx.frame_ctr = frame_ctr
    payload_bytes = tx._build_payload()

    print(f"Payload: {len(payload_bytes)} bytes")
    print(f"First 16 bytes: {payload_bytes[:16].hex()}")

    # Step 2: Polar encoding (should be identical)
    print("\n2. POLAR ENCODING")
    print("-" * 40)

    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype="u1"))
    data_bits = polar_enc(payload_bytes, N=1024, K=448)

    print(f"Payload bits: {len(payload_bits)} bits")
    print(f"Polar encoded: {len(data_bits)} bits")
    print(f"Data bits (first 32): {data_bits[:32]}")

    # Step 3: PN sequence generation (should be identical)
    print("\n3. PN SEQUENCE GENERATION")
    print("-" * 40)

    sec = SecureChannel(key)
    frame_len = 63 + len(data_bits)  # preamble + payload
    pn_full = sec.pn_bits(frame_ctr, frame_len)
    pn_payload = pn_full[63:]

    print(f"PN full length: {len(pn_full)} bits")
    print(f"PN payload length: {len(pn_payload)} bits")
    print(f"PN payload (first 32): {pn_payload[:32]}")

    # Step 4: Symbol generation and spreading (TX process)
    print("\n4. TX SYMBOL GENERATION AND SPREADING")
    print("-" * 40)

    preamble = np.array([1, 0, 1] * 21, dtype=np.uint8)[:63]
    preamble_symbols = 2 * preamble - 1  # No spreading for preamble

    data_symbols = 2 * data_bits.astype(np.float32) - 1  # Map to ±1
    pn_symbols = 2 * pn_payload.astype(np.float32) - 1  # Map to ±1
    spread_payload = data_symbols * pn_symbols  # Multiply (spreading)

    all_symbols = np.concatenate([preamble_symbols.astype(np.float32), spread_payload])

    print(f"Preamble symbols (first 16): {preamble_symbols[:16]}")
    print(f"Data symbols (first 16): {data_symbols[:16]}")
    print(f"PN symbols (first 16): {pn_symbols[:16]}")
    print(f"Spread payload (first 16): {spread_payload[:16]}")
    print(f"All symbols length: {len(all_symbols)}")

    # Step 5: Bandpass filtering (TX process)
    print("\n5. TX BANDPASS FILTERING")
    print("-" * 40)

    from rtwm.utils import choose_band
    band = choose_band(key, frame_ctr)
    b, a = butter_bandpass(*band, fs)

    filtered_symbols = lfilter(b, a, all_symbols)
    filtered_symbols /= np.sqrt(np.mean(filtered_symbols ** 2)) + 1e-12  # Normalize

    print(f"Band: {band}")
    print(f"Before filtering - mean: {np.mean(all_symbols):.6f}, std: {np.std(all_symbols):.6f}")
    print(f"After filtering - mean: {np.mean(filtered_symbols):.6f}, std: {np.std(filtered_symbols):.6f}")
    print(f"Filter gain at payload: ~{np.std(filtered_symbols) / np.std(all_symbols):.3f}")

    # Step 6: Actual TX output (for comparison)
    print("\n6. ACTUAL TX OUTPUT")
    print("-" * 40)

    tx_chips = tx._make_frame_chips()

    print(f"TX chips length: {len(tx_chips)}")
    print(f"TX chips - mean: {np.mean(tx_chips):.6f}, std: {np.std(tx_chips):.6f}")
    print(f"Match with manual filtering: {np.allclose(tx_chips, filtered_symbols, atol=1e-5)}")

    if not np.allclose(tx_chips, filtered_symbols, atol=1e-5):
        print("WARNING: TX output doesn't match manual process!")
        diff = tx_chips - filtered_symbols
        print(f"Max difference: {np.max(np.abs(diff)):.6f}")

    # Step 7: RX Processing - Preamble Detection
    print("\n7. RX PREAMBLE DETECTION")
    print("-" * 40)

    detector = WatermarkDetector(key)

    # Use the same filtering as TX for preamble template
    preamble_template = 2 * preamble.astype(np.float32) - 1
    filtered_template = lfilter(b, a, preamble_template)
    filtered_template /= np.sqrt(np.mean(filtered_template ** 2)) + 1e-12

    from scipy.signal import correlate
    correlation = correlate(tx_chips, filtered_template, mode='valid')

    peak_pos = np.argmax(correlation)
    peak_value = correlation[peak_pos]

    print(f"Preamble correlation peak: {peak_value:.3f} at position {peak_pos}")
    print(f"Expected peak position: ~{31} (half preamble length)")

    # Step 8: RX Frame Extraction
    print("\n8. RX FRAME EXTRACTION")
    print("-" * 40)

    peak_shift = (len(preamble) - 1) // 2
    frame_start = peak_pos - peak_shift
    extracted_frame = tx_chips[frame_start:frame_start + len(tx_chips)]

    if frame_start < 0 or frame_start + len(tx_chips) > len(tx_chips):
        print("WARNING: Frame extraction would go out of bounds!")
        extracted_frame = tx_chips  # Use full signal
        frame_start = 0

    print(f"Frame start: {frame_start}")
    print(f"Extracted frame length: {len(extracted_frame)}")

    # Step 9: RX Payload Extraction and Despreading
    print("\n9. RX PAYLOAD DESPREADING")
    print("-" * 40)

    rx_payload = extracted_frame[63:]  # Skip preamble
    print(f"RX payload length: {len(rx_payload)}")
    print(f"RX payload - mean: {np.mean(rx_payload):.6f}, std: {np.std(rx_payload):.6f}")

    # Despread with PN sequence
    min_len = min(len(rx_payload), len(pn_symbols))
    rx_payload_trimmed = rx_payload[:min_len]
    pn_symbols_trimmed = pn_symbols[:min_len]

    despread = rx_payload_trimmed * pn_symbols_trimmed

    print(f"Despread length: {len(despread)}")
    print(f"Despread - mean: {np.mean(despread):.6f}, std: {np.std(despread):.6f}")
    print(f"Despread (first 32): {despread[:32]}")
    print(f"Original data symbols (first 32): {data_symbols[:32]}")

    # Check correlation between despread and original data
    correlation_coeff = np.corrcoef(despread, data_symbols[:len(despread)])[0, 1]
    print(f"Correlation with original data symbols: {correlation_coeff:.6f}")

    # Step 10: LLR Calculation and Polar Decoding
    print("\n10. LLR CALCULATION AND POLAR DECODING")
    print("-" * 40)

    # Simple LLR: just scale the despread symbols
    llr_scale = 8.0
    llr = despread * llr_scale
    llr = np.clip(llr, -30.0, 30.0)

    # Pad to correct length if needed
    if len(llr) < 1024:
        llr_padded = np.zeros(1024, dtype=np.float32)
        llr_padded[:len(llr)] = llr
        llr = llr_padded

    print(f"LLR - mean: {np.mean(llr):.6f}, std: {np.std(llr):.6f}")
    print(f"LLR range: [{llr.min():.3f}, {llr.max():.3f}]")

    # Try polar decoding
    decoded_bytes = polar_dec(llr)

    if decoded_bytes is not None:
        print(f"Polar decoding: SUCCESS")
        print(f"Decoded length: {len(decoded_bytes)} bytes")
        print(f"Original payload: {payload_bytes[:16].hex()}")
        print(f"Decoded payload:  {decoded_bytes[:16].hex()}")
        print(f"Payloads match: {payload_bytes == decoded_bytes}")
    else:
        print(f"Polar decoding: FAILED")

        # Try different LLR scales
        print("Trying different LLR scales...")
        for scale in [1.0, 2.0, 4.0, 16.0, 32.0]:
            test_llr = np.clip(despread * scale, -30.0, 30.0)
            if len(test_llr) < 1024:
                test_llr_padded = np.zeros(1024, dtype=np.float32)
                test_llr_padded[:len(test_llr)] = test_llr
                test_llr = test_llr_padded

            test_decoded = polar_dec(test_llr)
            success = test_decoded is not None
            print(f"  Scale {scale:4.1f}: {'SUCCESS' if success else 'FAILED'}")

            if success and test_decoded == payload_bytes:
                print(f"    ✓ Perfect match with original payload!")
                return True

    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    print(f"Correlation with original data: {correlation_coeff:.6f}")
    print(f"Polar decoding success: {decoded_bytes is not None}")
    if decoded_bytes is not None:
        print(f"Payload match: {payload_bytes == decoded_bytes}")

    return decoded_bytes is not None and payload_bytes == decoded_bytes


if __name__ == "__main__":
    success = comprehensive_debug()
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILURE'}")