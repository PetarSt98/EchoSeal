import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
from rtwm.crypto import SecureChannel
from rtwm.utils import choose_band
from rtwm.polar_fast import encode as polar_enc


def debug_tx_rx_mismatch(key=b"\xAA" * 32):
    """
    Compare what TX sends vs what RX expects to receive.
    This will help identify the exact mismatch.
    """

    print("=" * 60)
    print("TX/RX SIGNAL COMPARISON DEBUG")
    print("=" * 60)

    # Create TX and RX
    tx = WatermarkEmbedder(key)
    rx = WatermarkDetector(key)
    sec = SecureChannel(key)

    # Generate one frame from TX
    frame_ctr = 0

    # Step 1: Get the same payload that TX would generate
    tx.frame_ctr = frame_ctr  # Set to known value
    payload_bytes = tx._build_payload()

    print(f"1. Payload: {len(payload_bytes)} bytes")
    print(f"   First 8 bytes: {payload_bytes[:8].hex()}")

    # Step 2: Encode with polar codes (same as TX)
    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype="u1"))
    print(f"2. Payload bits: {len(payload_bits)} bits")
    print(f"   First 16 bits: {payload_bits[:16]}")

    data_bits = polar_enc(payload_bytes, N=1024, K=448)
    print(f"3. Polar encoded: {len(data_bits)} bits")
    print(f"   First 16 bits: {data_bits[:16]}")

    # Step 3: Get PN sequence (same as TX)
    frame_len = 63 + len(data_bits)  # preamble + data
    pn_full = sec.pn_bits(frame_ctr, frame_len)
    pn_payload = pn_full[63:]  # Skip preamble part

    print(f"4. PN sequence: {len(pn_full)} bits total, {len(pn_payload)} for payload")
    print(f"   PN payload first 16: {pn_payload[:16]}")

    # Step 4: Check TX spreading operation
    preamble_sy = 2 * np.array([1, 0, 1] * 21, dtype=np.uint8)[:63] - 1
    payload_sy_tx = (2 * data_bits - 1) * (2 * pn_payload - 1)  # TX spreading

    print(f"5. TX spreading:")
    print(f"   Data symbols: {(2 * data_bits - 1)[:8]} (first 8)")
    print(f"   PN symbols: {(2 * pn_payload - 1)[:8]} (first 8)")
    print(f"   Spread result: {payload_sy_tx[:8]} (first 8)")

    # Step 5: What RX should do for despreading
    print(f"6. RX despreading check:")
    print(f"   To recover data: spread_signal * pn_symbols")
    recovered = payload_sy_tx * (2 * pn_payload - 1)
    print(f"   Recovered: {recovered[:8]} (first 8)")
    print(f"   Original data: {(2 * data_bits - 1)[:8]} (first 8)")
    print(f"   Match? {np.allclose(recovered, (2 * data_bits - 1))}")

    # Step 6: Generate actual TX signal
    tx_chips = tx._make_frame_chips()
    print(f"7. TX chips: {len(tx_chips)} samples")
    print(f"   Energy: {np.mean(tx_chips ** 2):.6f}")
    print(f"   Mean: {np.mean(tx_chips):.6f}")
    print(f"   Std: {np.std(tx_chips):.6f}")

    # Step 7: Check what happens when RX processes this
    print(f"8. RX processing simulation:")

    # Extract payload from TX signal (after bandpass filtering effects)
    payload_part = tx_chips[63:]  # Skip preamble
    print(f"   RX payload chips: mean={np.mean(payload_part):.6f}, std={np.std(payload_part):.6f}")

    # Normalize (as RX does)
    payload_normalized = payload_part / (np.sqrt(np.mean(payload_part ** 2)) + 1e-12)
    print(f"   After normalization: mean={np.mean(payload_normalized):.6f}, std={np.std(payload_normalized):.6f}")

    # Despread
    despread = payload_normalized * (2 * pn_payload - 1)
    print(f"   After despreading: mean={np.mean(despread):.6f}, std={np.std(despread):.6f}")
    print(f"   Range: [{despread.min():.6f}, {despread.max():.6f}]")

    # Check if despread signal correlates with original data
    data_symbols = (2 * data_bits - 1).astype(np.float32)
    correlation = np.corrcoef(despread, data_symbols)[0, 1]
    print(f"   Correlation with original data: {correlation:.6f}")

    if abs(correlation) < 0.1:
        print("   ⚠️  WARNING: Very low correlation suggests signal processing mismatch!")
    elif correlation > 0.8:
        print("   ✅ Good correlation - signal processing looks correct")
    else:
        print("   ⚠️  Moderate correlation - there might be some issues")

    return {
        'tx_chips': tx_chips,
        'payload_bits': payload_bits,
        'data_bits': data_bits,
        'pn_payload': pn_payload,
        'despread': despread,
        'correlation': correlation
    }


# Usage:
if __name__ == "__main__":
    results = debug_tx_rx_mismatch()