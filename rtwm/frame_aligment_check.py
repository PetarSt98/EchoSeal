import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
from rtwm.polar_fast import decode as polar_dec


def test_frame_counter_alignment(key=b"\xAA" * 32):
    """
    Test to ensure frame counter is properly aligned between TX and RX.
    """
    print("=" * 60)
    print("FRAME COUNTER ALIGNMENT TEST")
    print("=" * 60)

    # Create fresh instances
    tx = WatermarkEmbedder(key)
    rx = WatermarkDetector(key)

    # Test 1: Check initial frame counter
    print("\n1. INITIAL FRAME COUNTER CHECK")
    print("-" * 40)
    print(f"TX initial frame_ctr: {tx.frame_ctr}")

    # Generate a frame and check counter
    frame1 = tx._make_frame_chips()
    print(f"TX frame_ctr after first frame: {tx.frame_ctr}")

    # The frame was generated with frame_ctr=0, but now it's 1
    # So we need to test with frame_ctr=0

    # Test 2: Direct decode attempt with correct counter
    print("\n2. DECODE WITH FRAME_CTR=0")
    print("-" * 40)

    llr = rx._llr(frame1, 0)  # Use 0 because frame was generated with ctr=0
    blob = polar_dec(llr)

    if blob is not None:
        try:
            plain = rx.sec.open(blob)
            print(f"✅ Decode successful!")
            print(f"Payload prefix: {plain[:4]}")
            print(f"Embedded counter: {int.from_bytes(plain[4:8], 'big')}")
        except Exception as e:
            print(f"❌ Decrypt failed: {e}")
    else:
        print("❌ Polar decode failed")

    # Test 3: Try with different counters
    print("\n3. TRY DIFFERENT COUNTERS")
    print("-" * 40)

    for test_ctr in range(5):
        llr = rx._llr(frame1, test_ctr)
        blob = polar_dec(llr)

        if blob is not None:
            try:
                plain = rx.sec.open(blob)
                embedded_ctr = int.from_bytes(plain[4:8], 'big')
                print(f"Counter {test_ctr}: ✅ Decoded, embedded_ctr={embedded_ctr}")
                if embedded_ctr == test_ctr:
                    print(f"  → MATCH! Frame was generated with counter {test_ctr}")
            except:
                print(f"Counter {test_ctr}: ❌ Decrypt failed")
        else:
            print(f"Counter {test_ctr}: ❌ Polar decode failed")

    # Test 4: Generate multiple frames and test
    print("\n4. MULTIPLE FRAME TEST")
    print("-" * 40)

    tx2 = WatermarkEmbedder(key)  # Fresh instance
    rx2 = WatermarkDetector(key)

    frames = []
    expected_ctrs = []

    for i in range(3):
        print(f"\nGenerating frame {i}:")
        print(f"  TX frame_ctr before: {tx2.frame_ctr}")
        frame = tx2._make_frame_chips()
        frames.append(frame)
        expected_ctrs.append(i)  # Frame is generated with current counter value
        print(f"  TX frame_ctr after: {tx2.frame_ctr}")

    # Now try to decode each frame
    print("\nDecoding frames:")
    for i, (frame, expected_ctr) in enumerate(zip(frames, expected_ctrs)):
        print(f"\nFrame {i} (expected ctr={expected_ctr}):")

        # Try the expected counter
        llr = rx2._llr(frame, expected_ctr)
        blob = polar_dec(llr)

        if blob is not None:
            try:
                plain = rx2.sec.open(blob)
                embedded_ctr = int.from_bytes(plain[4:8], 'big')
                print(f"  ✅ Decoded with ctr={expected_ctr}, embedded_ctr={embedded_ctr}")
                if embedded_ctr == expected_ctr:
                    print(f"  → SUCCESS: Counters match!")
            except Exception as e:
                print(f"  ❌ Decrypt failed: {e}")
        else:
            print(f"  ❌ Polar decode failed with expected ctr={expected_ctr}")

    # Test 5: Verify PN sequence generation
    print("\n5. PN SEQUENCE VERIFICATION")
    print("-" * 40)

    # Generate PN from TX and RX for same counter
    test_ctr = 0
    pn_tx = tx.sec.pn_bits(test_ctr, 100)
    pn_rx = rx.sec.pn_bits(test_ctr, 100)

    print(f"PN sequences match: {np.array_equal(pn_tx, pn_rx)}")
    print(f"TX PN first 32 bits: {pn_tx[:32]}")
    print(f"RX PN first 32 bits: {pn_rx[:32]}")

    if not np.array_equal(pn_tx, pn_rx):
        # Find first difference
        for i in range(len(pn_tx)):
            if pn_tx[i] != pn_rx[i]:
                print(f"First difference at bit {i}: TX={pn_tx[i]}, RX={pn_rx[i]}")
                break

    return True


if __name__ == "__main__":
    test_frame_counter_alignment()