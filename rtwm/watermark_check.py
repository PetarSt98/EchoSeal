import numpy as np
from scipy.signal import lfilter
from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.detector import WatermarkDetector, PREAMBLE
from rtwm.crypto import SecureChannel
from rtwm.utils import choose_band, butter_bandpass
from rtwm.polar_fast import encode as polar_enc, decode as polar_dec


class FixedPayloadEmbedder(WatermarkEmbedder):
    """Embedder that uses a fixed payload for testing."""

    def __init__(self, key32: bytes, params: TxParams | None = None):
        super().__init__(key32, params)
        # Use a fixed seed for reproducibility
        self.fixed_random = b'\x01\x23\x45\x67\x89\xAB\xCD\xEF' * 3  # 24 bytes

    def _build_payload(self) -> bytes:
        """Build payload with fixed random bytes instead of secrets.token_bytes."""
        PLAINTEXT_LEN = 27

        meta = (
                b"ESAL"
                + self.frame_ctr.to_bytes(4, "big")
                + self.fixed_random[:19]  # Use fixed bytes instead of random
        )

        assert len(meta) == PLAINTEXT_LEN
        blob = self.sec.seal(meta)
        assert len(blob) == 55
        return blob


def test_with_fixed_payload(key=b"\xAA" * 32):
    """Test with fixed payload to isolate the issue."""
    print("=" * 80)
    print("FIXED PAYLOAD TEST")
    print("=" * 80)

    # Create embedder with fixed payload
    tx = FixedPayloadEmbedder(key)
    rx = WatermarkDetector(key)

    # Generate frame
    print("\n1. GENERATE FRAME WITH FIXED PAYLOAD")
    print("-" * 40)

    tx.frame_ctr = 0
    frame = tx._make_frame_chips()
    print(f"Frame generated, length: {len(frame)}")
    print(f"Frame stats: mean={np.mean(frame):.6f}, std={np.std(frame):.6f}")

    # Try to decode
    print("\n2. ATTEMPT DECODE")
    print("-" * 40)

    result = rx.verify_raw_frame(frame)
    print(f"Direct decode result: {'PASS' if result else 'FAIL'}")

    # Manual decode attempt
    print("\n3. MANUAL DECODE ATTEMPT")
    print("-" * 40)

    llr = rx._llr(frame, 0)
    print(f"LLR stats: mean={np.mean(llr):.3f}, std={np.std(llr):.3f}")

    blob = polar_dec(llr)
    if blob is not None:
        print("Polar decode successful")
        try:
            plain = rx.sec.open(blob)
            print(f"Decrypt successful!")
            print(f"Payload: {plain[:16].hex()}")
            print(f"Prefix: {plain[:4]}")
            print(f"Counter: {int.from_bytes(plain[4:8], 'big')}")
        except Exception as e:
            print(f"Decrypt failed: {e}")
    else:
        print("Polar decode failed")

    # Let's manually trace through the entire process
    print("\n4. MANUAL TRACE THROUGH PROCESS")
    print("-" * 40)

    # Get the payload that was used
    tx.frame_ctr = 0  # Reset
    payload_bytes = tx._build_payload()
    print(f"Payload (55 bytes): {payload_bytes[:16].hex()}...")

    # Polar encode
    data_bits = polar_enc(payload_bytes, N=1024, K=448)
    print(f"Polar encoded bits: {data_bits[:32]}")

    # Get PN sequence
    pn_full = tx.sec.pn_bits(0, 1087)
    pn_payload = pn_full[63:]

    # Create symbols
    preamble_symbols = 2.0 * PREAMBLE.astype(np.float32) - 1.0
    data_symbols = 2.0 * data_bits.astype(np.float32) - 1.0
    pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

    # Spread
    spread_payload = data_symbols * pn_symbols

    # Concatenate
    all_symbols = np.concatenate((preamble_symbols, spread_payload))

    # Filter
    band = choose_band(key, 0)
    b, a = butter_bandpass(*band, 48000)
    filtered = lfilter(b, a, all_symbols)

    # Normalize
    energy = np.mean(filtered ** 2)
    if energy > 1e-12:
        filtered /= np.sqrt(energy)

    print(f"\nManual reconstruction matches frame: {np.allclose(filtered, frame, rtol=1e-4)}")

    if not np.allclose(filtered, frame, rtol=1e-4):
        diff = np.abs(filtered - frame)
        max_diff_idx = np.argmax(diff)
        print(f"Max difference: {diff[max_diff_idx]:.6f} at index {max_diff_idx}")
        print(f"Manual: {filtered[max_diff_idx]:.6f}")
        print(f"Frame:  {frame[max_diff_idx]:.6f}")

    # Now let's check despreading with our manual frame
    print("\n5. DESPREAD MANUAL FRAME")
    print("-" * 40)

    payload_part = filtered[63:]
    despread = payload_part * pn_symbols

    print(f"Despread stats: mean={np.mean(despread):.3f}, std={np.std(despread):.3f}")

    # Check bit errors
    detected_bits = (despread > 0).astype(int)
    original_bits = (data_symbols > 0).astype(int)
    bit_errors = np.sum(detected_bits != original_bits)
    ber = bit_errors / len(original_bits)

    print(f"Bit errors: {bit_errors}/{len(original_bits)} (BER: {ber:.3f})")

    # If BER is good, try decoding
    if ber < 0.1:
        print("\nBER is good, attempting decode of manual frame...")
        llr_manual = rx._llr(filtered, 0)
        blob_manual = polar_dec(llr_manual)
        if blob_manual is not None:
            try:
                plain_manual = rx.sec.open(blob_manual)
                print(f"✅ Manual frame decoded successfully!")
            except:
                print("❌ Manual frame decrypt failed")
        else:
            print("❌ Manual frame polar decode failed")

    return frame, filtered


def test_embedder_internals(key=b"\xAA" * 32):
    """Test embedder internals to understand the mismatch."""
    print("\n" + "=" * 80)
    print("EMBEDDER INTERNALS TEST")
    print("=" * 80)

    # Create a fresh embedder
    tx = WatermarkEmbedder(key)

    # Capture the state before calling _make_frame_chips
    print("\n1. BEFORE _make_frame_chips()")
    print("-" * 40)
    print(f"frame_ctr: {tx.frame_ctr}")

    # Now let's trace through _make_frame_chips step by step
    frame_ctr_used = tx.frame_ctr
    band = choose_band(tx.sec.master_key, frame_ctr_used)
    b, a = butter_bandpass(*band, tx.p.fs)

    # Get payload
    payload = tx._build_payload()
    print(f"Payload generated: {len(payload)} bytes")

    # The actual bits that will be printed
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype="u1"))
    print(f"[TX] bits: {payload_bits[:16]}... len={len(payload_bits)}")

    # Polar encode
    data_bits = polar_enc(payload, N=tx.p.N, K=tx.p.K)

    # Get the actual frame
    tx.frame_ctr = frame_ctr_used  # Reset to same value
    actual_frame = tx._make_frame_chips()

    print(f"\n2. AFTER _make_frame_chips()")
    print("-" * 40)
    print(f"frame_ctr: {tx.frame_ctr}")
    print(f"Frame was generated with counter: {frame_ctr_used}")

    return actual_frame


if __name__ == "__main__":
    # Test with fixed payload
    frame, manual = test_with_fixed_payload()

    # Test embedder internals
    test_embedder_internals()