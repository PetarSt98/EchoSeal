import numpy as np
from scipy.signal import lfilter, filtfilt, correlate
from scipy.fft import fft, fftfreq

from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector, PREAMBLE
from rtwm.crypto import SecureChannel
from rtwm.utils import choose_band, butter_bandpass
from rtwm.polar_fast import encode as polar_enc


def deep_debug_analysis(key=b"\xAA" * 32):
    """
    Deep debugging to find exact mismatch between TX and RX.
    """
    print("=" * 80)
    print("DEEP DEBUG: TX/RX MISMATCH ANALYSIS")
    print("=" * 80)

    # Create instances
    tx = WatermarkEmbedder(key)
    rx = WatermarkDetector(key)
    sec = SecureChannel(key)

    # Parameters
    frame_ctr = 0
    band = choose_band(key, frame_ctr)
    b, a = butter_bandpass(*band, tx.p.fs)

    print(f"Band: {band}, Filter order: {len(b) - 1}")

    # Step 1: Generate TX frame components separately
    print("\n1. TX FRAME GENERATION (Step by Step)")
    print("-" * 40)

    # Build payload
    tx.frame_ctr = frame_ctr
    payload_bytes = tx._build_payload()
    data_bits = polar_enc(payload_bytes, N=1024, K=448)

    # Generate PN sequence
    frame_len = 63 + 1024
    pn_full = sec.pn_bits(frame_ctr, frame_len)
    pn_preamble = pn_full[:63]
    pn_payload = pn_full[63:]

    # Create symbols BEFORE filtering
    preamble_symbols = 2.0 * PREAMBLE.astype(np.float32) - 1.0
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0
    pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

    # Spread payload
    spread_payload = data_symbols * pn_symbols

    # Concatenate (before filtering)
    symbols_unfiltered = np.concatenate((preamble_symbols, spread_payload))

    print(f"Preamble symbols: mean={np.mean(preamble_symbols):.3f}, std={np.std(preamble_symbols):.3f}")
    print(f"Spread payload: mean={np.mean(spread_payload):.3f}, std={np.std(spread_payload):.3f}")
    print(f"Combined symbols: len={len(symbols_unfiltered)}")

    # Step 2: Apply filtering (as TX does)
    print("\n2. FILTERING ANALYSIS")
    print("-" * 40)

    # Filter the complete symbol sequence
    chips_filtered = lfilter(b, a, symbols_unfiltered)

    # Normalize to unit energy (as TX does)
    energy = np.mean(chips_filtered ** 2)
    if energy > 1e-12:
        chips_filtered /= np.sqrt(energy)

    print(f"Filtered chips: mean={np.mean(chips_filtered):.6f}, std={np.std(chips_filtered):.3f}")
    print(f"Energy after normalization: {np.mean(chips_filtered ** 2):.3f}")

    # Step 3: Compare with actual TX output
    print("\n3. COMPARISON WITH ACTUAL TX OUTPUT")
    print("-" * 40)

    tx.frame_ctr = frame_ctr  # Reset counter
    tx_chips_actual = tx._make_frame_chips()

    chips_match = np.allclose(chips_filtered, tx_chips_actual, rtol=1e-5)
    print(f"Our reconstruction matches TX output: {chips_match}")
    if not chips_match:
        max_diff = np.max(np.abs(chips_filtered - tx_chips_actual))
        print(f"Max difference: {max_diff}")

    # Step 4: Analyze preamble correlation issue
    print("\n4. PREAMBLE CORRELATION ANALYSIS")
    print("-" * 40)

    # Extract preamble from filtered signal
    preamble_in_signal = chips_filtered[:63]

    # Create different preamble templates
    template_unfiltered = preamble_symbols
    template_filtered = lfilter(b, a, preamble_symbols)
    template_filtered_norm = template_filtered / np.sqrt(np.mean(template_filtered ** 2))

    # Check correlations
    corr1 = np.dot(preamble_in_signal, template_unfiltered) / (
        np.sqrt(np.sum(preamble_in_signal ** 2) * np.sum(template_unfiltered ** 2))
    )
    corr2 = np.dot(preamble_in_signal, template_filtered) / (
        np.sqrt(np.sum(preamble_in_signal ** 2) * np.sum(template_filtered ** 2))
    )
    corr3 = np.dot(preamble_in_signal, template_filtered_norm) / (
        np.sqrt(np.sum(preamble_in_signal ** 2) * np.sum(template_filtered_norm ** 2))
    )

    print(f"Correlation with unfiltered template: {corr1:.3f}")
    print(f"Correlation with filtered template: {corr2:.3f}")
    print(f"Correlation with filtered+normalized template: {corr3:.3f}")

    # Step 5: Test despreading on properly aligned frame
    print("\n5. DESPREADING TEST (Properly Aligned)")
    print("-" * 40)

    # Extract payload from filtered signal
    payload_in_signal = chips_filtered[63:63 + 1024]

    # Despread
    despread = payload_in_signal * pn_symbols

    # Check bit errors
    detected_bits = (despread > 0).astype(int)
    original_bits = (data_bits > 0).astype(int)
    bit_errors = np.sum(detected_bits != original_bits)
    ber = bit_errors / len(data_bits)

    print(f"Despread stats: mean={np.mean(despread):.3f}, std={np.std(despread):.3f}")
    print(f"Bit errors: {bit_errors}/{len(data_bits)} (BER: {ber:.3f})")

    # Correlation
    corr_despread = np.corrcoef(despread, data_symbols)[0, 1]
    print(f"Correlation between despread and original: {corr_despread:.3f}")

    # Step 6: Frame search simulation
    print("\n6. FRAME SEARCH SIMULATION")
    print("-" * 40)

    # Simulate what detector does
    # Use the properly filtered template
    template_for_search = lfilter(b, a, preamble_symbols)

    # Try correlation at different positions
    print("Correlation at different offsets:")
    for offset in [0, 1, 2, 3, 4, 5, 10, 20, 50]:
        if offset + 63 <= len(chips_filtered):
            segment = chips_filtered[offset:offset + 63]
            seg_energy = np.sum(segment ** 2)
            temp_energy = np.sum(template_for_search ** 2)
            if seg_energy > 1e-12 and temp_energy > 1e-12:
                corr = np.dot(segment, template_for_search) / np.sqrt(seg_energy * temp_energy)
                print(f"  Offset {offset}: {corr:.3f}")

    # Step 7: Check if the issue is with the entire frame pipeline
    print("\n7. FULL DETECTION PIPELINE TEST")
    print("-" * 40)

    # Test with our reconstructed signal
    class DebugDetector(WatermarkDetector):
        def test_frame_direct(self, frame, frame_id):
            """Bypass all detection logic and test LLR directly."""
            return self._llr(frame, frame_id)

    debug_rx = DebugDetector(key)
    llr = debug_rx.test_frame_direct(chips_filtered, frame_ctr)

    print(f"LLR stats: mean={np.mean(llr):.3f}, std={np.std(llr):.3f}")
    print(f"LLR range: [{np.min(llr):.3f}, {np.max(llr):.3f}]")

    # Check if we can decode
    from rtwm.polar_fast import decode as polar_dec
    blob = polar_dec(llr)
    if blob is not None:
        try:
            plain = sec.open(blob)
            print(f"✅ Decoding successful! Payload starts with: {plain[:8].hex()}")
        except Exception as e:
            print(f"❌ Decrypt failed: {type(e).__name__}")
    else:
        print("❌ Polar decode failed")

    # Step 8: The smoking gun - check the actual TX implementation
    print("\n8. TX IMPLEMENTATION CHECK")
    print("-" * 40)

    # Let's trace through TX's actual frame generation
    tx.frame_ctr = frame_ctr
    tx._chip_buf = np.empty(0, dtype=np.float32)  # Reset buffer

    # Process with empty input to force frame generation
    dummy_input = np.zeros(2048)
    output = tx.process(dummy_input)

    # The chips should be in the buffer
    generated_chips = output - dummy_input  # Extract just the watermark
    alpha = 10 ** (tx.p.target_rel_db / 20.0)  # Convert dB to linear
    generated_chips = generated_chips / alpha  # Undo scaling

    print(
        f"TX process() output stats: mean={np.mean(generated_chips[:1087]):.6f}, std={np.std(generated_chips[:1087]):.3f}")

    return {
        'chips_filtered': chips_filtered,
        'tx_chips_actual': tx_chips_actual,
        'data_symbols': data_symbols,
        'pn_symbols': pn_symbols,
        'despread': despread,
        'llr': llr,
        'bit_errors': bit_errors,
        'correlation': corr_despread
    }


if __name__ == "__main__":
    results = deep_debug_analysis()