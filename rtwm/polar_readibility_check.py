#!/usr/bin/env python3
"""
Verify the polar code reliability sequence is correct.
"""

import numpy as np

def test_q_nmax_check():
    from rtwm.reliability_polar_bits import Q_Nmax
    import numpy as np

    reliability = np.array(list(map(int, Q_Nmax.split())))
    print(f"First 5 indices in Q_Nmax: {reliability[:5]}")
    print(f"Last 5 indices in Q_Nmax: {reliability[-5:]}")

    # Check if 0 is in first half or second half
    pos_of_zero = np.where(reliability == 0)[0][0]
    print(f"Position of index 0 in sequence: {pos_of_zero}")
    print(f"Position of index 1023 in sequence: {np.where(reliability == 1023)[0][0]}")
    return True

def verify_reliability_sequence():
    """Check if Q_Nmax is properly sorted and used."""
    print("=" * 60)
    print("POLAR CODE RELIABILITY VERIFICATION")
    print("=" * 60)

    from rtwm.reliability_polar_bits import Q_Nmax

    # Parse the reliability sequence
    reliability = np.array(list(map(int, Q_Nmax.split())))

    print(f"Reliability sequence length: {len(reliability)}")
    print(f"Expected length: 1024")

    if len(reliability) != 1024:
        print("✗ Wrong sequence length!")
        return False

    # Check if all indices 0-1023 are present
    sorted_rel = np.sort(reliability)
    expected = np.arange(1024)

    if np.array_equal(sorted_rel, expected):
        print("✓ All indices 0-1023 are present")
    else:
        print("✗ Missing or duplicate indices!")
        return False

    # The sequence should be ordered by reliability (least to most reliable)
    # In Q_Nmax, the first elements are the LEAST reliable (frozen)
    # and the last elements are the MOST reliable (information)

    print(f"\nFirst 10 indices (least reliable, frozen): {reliability[:10]}")
    print(f"Last 10 indices (most reliable, info): {reliability[-10:]}")

    # For K=448, we want the 448 most reliable positions
    K = 448
    frozen_mask = np.ones(1024, dtype=bool)
    frozen_mask[reliability[:K]] = False  # This is WRONG!

    print(f"\nCurrent implementation unfreezes first {K} values from Q_Nmax")
    print(f"These are positions: {reliability[:10]}... (least reliable!)")

    # The correct way:
    frozen_mask_correct = np.ones(1024, dtype=bool)
    frozen_mask_correct[reliability[-K:]] = False  # Take LAST K (most reliable)

    print(f"\nCorrect implementation should unfreeze last {K} values from Q_Nmax")
    print(f"These are positions: {reliability[-10:]}... (most reliable!)")

    print("\n✗ BUG FOUND: Frozen bit selection is INVERTED!")
    print("  Currently using LEAST reliable bits for information")
    print("  Should use MOST reliable bits for information")

    return False


def test_corrected_polar():
    """Test with corrected frozen bit selection."""
    print("\n" + "=" * 60)
    print("TEST WITH CORRECTED FROZEN BITS")
    print("=" * 60)

    from rtwm.reliability_polar_bits import Q_Nmax

    N = 1024
    K = 448

    # Parse reliability sequence
    reliability = np.array(list(map(int, Q_Nmax.split())))

    # CORRECT: Take the K most reliable positions (last K in Q_Nmax)
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[reliability[-K:]] = False  # Most reliable positions

    print(f"Number of frozen bits: {np.sum(frozen_mask)}")
    print(f"Number of info bits: {np.sum(~frozen_mask)}")

    # Show which positions are used for information
    info_positions = np.where(~frozen_mask)[0]
    print(f"First 10 info positions: {info_positions[:10]}")
    print(f"Last 10 info positions: {info_positions[-10:]}")

    # These should be high-reliability positions
    # Typically includes positions like 512, 768, 896, etc.

    return True


if __name__ == "__main__":
    print("qnmax")
    print("qnmax")
    verify_reliability_sequence()
    test_corrected_polar()