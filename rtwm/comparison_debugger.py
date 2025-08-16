#!/usr/bin/env python3
"""
Comprehensive debugging tool to identify TX/RX mismatches.
This will help us find exactly where the signal processing goes wrong.
"""

import numpy as np
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector, PRE_BITS, PRE_L, FRAME_LEN
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
    tx._build_payload = lambda: payload_bytes  # Monkey-patch so embedder uses the same payload

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
    pn_full = sec.pn_bits(frame_ctr, FRAME_LEN)
    pn_payload = pn_full[PRE_L:]

    print(f"PN full length: {len(pn_full)} bits")
    print(f"PN payload length: {len(pn_payload)} bits")
    print(f"PN payload (first 32): {pn_payload[:32]}")

    # Step 4: Symbol generation and spreading (TX process)
    print("\n4. TX SYMBOL GENERATION AND SPREADING")
    print("-" * 40)

    preamble_symbols = 2.0 * PRE_BITS.astype(np.float32) - 1.0

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
    sec = SecureChannel(key)
    band_key = getattr(sec, "band_key", key)
    band = choose_band(band_key, frame_ctr)
    b, a = butter_bandpass(*band, fs, order=4)
    zi_len = max(len(a), len(b)) - 1
    zi = np.zeros(zi_len, dtype=np.result_type(a, b, all_symbols))
    filtered_symbols, _ = lfilter(b, a, all_symbols, zi=zi)
    start = max(16, zi_len)  # same steady-state window as TX
    steady = filtered_symbols[start:] if filtered_symbols.size > start else filtered_symbols
    energy = float(np.mean(steady ** 2))
    if energy > 1e-12:
        filtered_symbols /= np.sqrt(energy)
    filtered_symbols = filtered_symbols.astype(np.float32, copy=False)
    print(f"Band: {band}")
    print(f"Before filtering - mean: {np.mean(all_symbols):.6f}, std: {np.std(all_symbols):.6f}")
    print(f"After filtering - mean: {np.mean(filtered_symbols):.6f}, std: {np.std(filtered_symbols):.6f}")
    print(f"Filter gain at payload: ~{np.std(filtered_symbols) / np.std(all_symbols):.3f}")

    # Step 6: Actual TX output (for comparison)
    print("\n6. ACTUAL TX OUTPUT")
    print("-" * 40)

    tx_chips = tx._make_frame_chips()

    print("TX chips dtype:", tx_chips.dtype, "filtered_symbols dtype:", filtered_symbols.dtype)
    print("TX chips max:", np.max(np.abs(tx_chips)), "filtered_symbols max:", np.max(np.abs(filtered_symbols)))
    print("First 10 samples TX chips:", tx_chips[:10])
    print("First 10 samples filtered_symbols:", filtered_symbols[:10])

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
    preamble_template = 2.0 * PRE_BITS.astype(np.float32) - 1.0
    filtered_template = lfilter(b, a, preamble_template)
    filtered_template /= (np.sqrt(np.sum(filtered_template ** 2)) + 1e-12)

    from scipy.signal import correlate
    win_energy = np.sqrt(np.convolve(tx_chips * tx_chips,
                                     np.ones(filtered_template.size, dtype=np.float32),
                                     mode='valid')) + 1e-12
    corr = correlate(tx_chips, filtered_template, mode='valid') / win_energy
    peak_pos = int(np.argmax(corr))
    peak_value = float(corr[peak_pos])

    print(f"Preamble correlation peak: {peak_value:.3f} at position {peak_pos}")
    print("Expected peak position: 0 (start-of-frame)")

    # Step 8: RX Frame Extraction
    print("\n8. RX FRAME EXTRACTION")
    print("-" * 40)

    frame_start = peak_pos
    extracted_frame = tx_chips[frame_start:frame_start + FRAME_LEN]

    if frame_start < 0 or frame_start + FRAME_LEN > len(tx_chips):
        print("WARNING: Frame extraction would go out of bounds!")
        extracted_frame = tx_chips  # Use full signal
        frame_start = 0

    print(f"Frame start: {frame_start}")
    print(f"Extracted frame length: {len(extracted_frame)}")

    # Step 9: RX Payload Extraction and Despreading
    print("\n9. RX PAYLOAD DESPREADING")
    print("-" * 40)

    rx_payload = extracted_frame[PRE_L:]  # Skip preamble
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

    # Step 10: LLR CALCULATION AND POLAR DECODING
    print("\n10. LLR CALCULATION AND POLAR DECODING")
    print("-" * 40)

    sec_tx = SecureChannel(key)
    pn_tx = sec_tx.pn_bits(frame_ctr, FRAME_LEN)

    sec_rx = detector.sec
    pn_rx = sec_rx.pn_bits(frame_ctr, FRAME_LEN)

    print("PN identical:", np.array_equal(pn_tx, pn_rx))

    # Use the detector's LLR exactly as in production RX
    llr = detector._llr(tx_chips, frame_ctr)  # float32, length 1024
    print(f"LLR - mean: {llr.mean():.6f}, std: {llr.std():.6f}, "
          f"range: [{llr.min():.3f}, {llr.max():.3f}]")
    hd = np.mean(((llr > 0).astype(np.uint8) ^ data_bits).astype(np.float32))
    print(f"Hard-decision BER vs codeword: {hd:.3f}")
    ber = np.mean(((llr > 0).astype(np.uint8) ^ data_bits).astype(np.float32))
    print(f"Hard-decision BER vs codeword: {ber:.3f}")

    # Decode; if CRC fails (None), try a sign-flip once as a diagnostic
    decoded_bytes = polar_dec(llr)
    if decoded_bytes is None:
        decoded_bytes = polar_dec(-llr)

    print("\n[DEBUG] --- POLAR DECODER BYTEWISE CHECK ---")
    if decoded_bytes is None:
        print("[DEBUG] Decoder failed CRC (no bytes returned).")
        success = False
    else:
        print(f"[DEBUG] Encoded payload bytes: {payload_bytes.hex()}")
        print(f"[DEBUG] Decoded bytes:        {decoded_bytes.hex()}")
        success = (decoded_bytes == payload_bytes)
        if not success:
            print("[DEBUG] Payloads do NOT match, byte-wise differences:")
            for i, (a, b) in enumerate(zip(payload_bytes, decoded_bytes)):
                if a != b:
                    print(f"  Byte {i}: encoded={a:02x} decoded={b:02x}")
            if len(payload_bytes) != len(decoded_bytes):
                print(f"[DEBUG] Length mismatch: encoded {len(payload_bytes)} vs decoded {len(decoded_bytes)}")
            print(f"[DEBUG] Last byte of encoded: {payload_bytes[-1]:02x}")
            print(f"[DEBUG] Last byte of decoded: {decoded_bytes[-1]:02x}")

    print("\n" + "=" * 80)
    print("DEBUG SUMMARY")
    print("=" * 80)
    print(f"Polar decoding success: {decoded_bytes is not None}")
    if decoded_bytes is not None:
        print(f"Payload match: {success}")

    best = (-1.0, 0)
    sym = 2 * data_bits.astype(np.int8) - 1
    for k in range(-64, 65):
        s = np.sign(np.roll(llr, k)).astype(np.int8)
        corr = float(np.mean(s[:len(sym)] * sym))
        if corr > best[0]:
            best = (corr, k)
    print(f"Best corr vs codeword: {best[0]:.3f} at shift {best[1]} samples")

    return (decoded_bytes is not None) and success


if __name__ == "__main__":
    success = comprehensive_debug()
    print(f"\nOverall result: {'SUCCESS' if success else 'FAILURE'}")