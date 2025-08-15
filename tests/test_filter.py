import numpy as np
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt
from rtwm.utils import butter_bandpass


def test_filter_impact():
    """
    Test to understand how the bandpass filter affects the spread signal.
    """
    print("=" * 60)
    print("FILTER IMPACT TEST")
    print("=" * 60)

    # Parameters
    fs = 48000
    band = (8000, 10000)
    b, a = butter_bandpass(*band, fs, order=4)

    # Test 1: Filter frequency response
    print("\n1. FILTER FREQUENCY RESPONSE")
    print("-" * 40)

    w, h = freqz(b, a, worN=8000)
    freq = w * fs / (2 * np.pi)

    # Find -3dB points
    mag_db = 20 * np.log10(np.abs(h))
    passband_idx = np.where((freq >= band[0]) & (freq <= band[1]))[0]
    passband_mag = np.mean(mag_db[passband_idx])

    print(f"Passband average magnitude: {passband_mag:.1f} dB")
    print(f"Filter order: {len(b) - 1}")

    # Test 2: Impact on spread spectrum signal
    print("\n2. SPREAD SPECTRUM SIGNAL TEST")
    print("-" * 40)

    # Create a short spread sequence
    np.random.seed(42)
    data = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
    pn = np.random.choice([-1, 1], size=8).astype(np.float32)
    spread = data * pn

    print(f"Original data: {data}")
    print(f"PN sequence:   {pn}")
    print(f"Spread signal: {spread}")

    # Apply filter
    filtered = lfilter(b, a, spread)

    print(f"\nAfter filtering:")
    print(f"Filtered:      {filtered}")

    # Try to despread
    despread = filtered * pn
    print(f"\nDespread:      {despread}")
    print(f"Expected:      {data}")

    # Check correlation
    corr = np.corrcoef(despread[:len(data)], data)[0, 1]
    print(f"Correlation:   {corr:.3f}")

    # Test 3: Longer sequence
    print("\n3. LONGER SEQUENCE TEST")
    print("-" * 40)

    # Create longer sequences
    N = 1024
    data_long = np.random.choice([-1, 1], size=N).astype(np.float32)
    pn_long = np.random.choice([-1, 1], size=N).astype(np.float32)
    spread_long = data_long * pn_long

    # Apply filter
    filtered_long = lfilter(b, a, spread_long)

    # Normalize
    filtered_norm = filtered_long / np.sqrt(np.mean(filtered_long ** 2))

    # Despread
    despread_long = filtered_norm * pn_long

    # Check bit errors
    detected = (despread_long > 0).astype(int)
    original = (data_long > 0).astype(int)
    bit_errors = np.sum(detected != original)
    ber = bit_errors / N

    print(f"Bit errors: {bit_errors}/{N} (BER: {ber:.3f})")

    # Test 4: Understand the filter delay
    print("\n4. FILTER DELAY ANALYSIS")
    print("-" * 40)

    # Create impulse
    impulse = np.zeros(100)
    impulse[50] = 1

    # Filter impulse
    impulse_response = lfilter(b, a, impulse)

    # Find peak
    peak_idx = np.argmax(np.abs(impulse_response))
    delay = peak_idx - 50

    print(f"Filter delay: {delay} samples")

    # Test 5: Try compensating for delay
    print("\n5. DELAY COMPENSATION TEST")
    print("-" * 40)

    # Shift the filtered signal to compensate for delay
    if delay > 0:
        filtered_compensated = np.roll(filtered_long, -delay)

        # Despread with compensation
        despread_comp = filtered_compensated * pn_long

        # Check bit errors
        detected_comp = (despread_comp > 0).astype(int)
        bit_errors_comp = np.sum(detected_comp != original)
        ber_comp = bit_errors_comp / N

        print(f"With delay compensation:")
        print(f"Bit errors: {bit_errors_comp}/{N} (BER: {ber_comp:.3f})")

    # Test 6: Check if issue is filter phase response
    print("\n6. PHASE RESPONSE CHECK")
    print("-" * 40)

    # For spread spectrum, we need linear phase
    # Butterworth doesn't have linear phase, which could be the issue

    # Test with simple alternating pattern
    test_pattern = np.array([1, -1] * 50, dtype=np.float32)
    filtered_pattern = lfilter(b, a, test_pattern)

    # Check if phase is inverted
    start_idx = 20  # Skip transient
    orig_seg = test_pattern[start_idx:start_idx + 20]
    filt_seg = filtered_pattern[start_idx:start_idx + 20]

    phase_corr = np.corrcoef(orig_seg, filt_seg)[0, 1]
    print(f"Phase correlation: {phase_corr:.3f}")

    if phase_corr < -0.5:
        print("WARNING: Filter appears to invert phase!")

    return ber, delay


if __name__ == "__main__":
    ber, delay = test_filter_impact()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    if ber > 0.4:
        print("The Butterworth bandpass filter is destroying the spread spectrum signal!")
        print("This is likely due to:")
        print("1. Non-linear phase response distorting the chip timing")
        print("2. Filter transients at symbol boundaries")
        print("3. Narrow bandwidth limiting the spread spectrum signal")
        print("\nPossible solutions:")
        print("1. Use a linear-phase FIR filter instead")
        print("2. Apply spreading AFTER frequency translation")
        print("3. Use a wider bandwidth filter")
        print("4. Use a matched filter approach in the receiver")