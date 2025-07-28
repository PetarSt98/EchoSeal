import numpy as np
from scipy.signal import lfilter
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector, PREAMBLE
from rtwm.utils import choose_band, butter_bandpass


def final_diagnosis(key=b"\xAA" * 32):
    """Final diagnosis of the watermarking system."""
    print("=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)

    # Create fresh instances
    tx = WatermarkEmbedder(key)
    rx = WatermarkDetector(key)

    # Generate a frame
    print("\n1. GENERATE TEST FRAME")
    print("-" * 40)
    tx.frame_ctr = 0
    frame = tx._make_frame_chips()
    print(f"Frame length: {len(frame)}")
    print(f"Frame energy: {np.mean(frame ** 2):.6f}")

    # Get the filter
    band = choose_band(key, 0)
    b, a = butter_bandpass(*band, 48000)

    # Check what the preamble looks like after filtering
    print("\n2. PREAMBLE ANALYSIS")
    print("-" * 40)

    # Create preamble symbols
    preamble_symbols = 2.0 * PREAMBLE.astype(np.float32) - 1.0

    # Method 1: Filter just preamble (wrong)
    preamble_filtered_alone = lfilter(b, a, preamble_symbols)

    # Method 2: Filter as part of frame (right)
    dummy_frame = np.zeros(1087, dtype=np.float32)
    dummy_frame[:63] = preamble_symbols
    dummy_filtered = lfilter(b, a, dummy_frame)
    preamble_filtered_inframe = dummy_filtered[:63]

    print(f"Preamble alone energy: {np.mean(preamble_filtered_alone ** 2):.6f}")
    print(f"Preamble in-frame energy: {np.mean(preamble_filtered_inframe ** 2):.6f}")
    print(f"Ratio: {np.mean(preamble_filtered_inframe ** 2) / np.mean(preamble_filtered_alone ** 2):.3f}")

    # Check the actual preamble in the TX frame
    tx_preamble = frame[:63]
    print(f"TX preamble energy: {np.mean(tx_preamble ** 2):.6f}")

    # The issue might be that the whole frame is normalized together
    # So the preamble and payload have different scales

    print("\n3. ENERGY DISTRIBUTION IN FRAME")
    print("-" * 40)

    # Check energy in different parts of the frame
    preamble_part = frame[:63]
    payload_part = frame[63:]

    print(f"Preamble energy: {np.mean(preamble_part ** 2):.6f}")
    print(f"Payload energy: {np.mean(payload_part ** 2):.6f}")
    print(f"Energy ratio (payload/preamble): {np.mean(payload_part ** 2) / np.mean(preamble_part ** 2):.3f}")

    # This energy imbalance might be the issue!
    # If the payload has much higher energy than the preamble after filtering,
    # then when we normalize the whole frame, the preamble gets scaled down

    print("\n4. FILTER TRANSIENT ANALYSIS")
    print("-" * 40)

    # The filter has a transient response that affects the beginning of the signal
    # Let's check how this affects different parts

    # Create a test signal with constant symbols
    test_const = np.ones(1087, dtype=np.float32)
    test_filtered = lfilter(b, a, test_const)

    print("Filter transient effect (constant input):")
    print(f"Sample 0-10: {test_filtered[:10]}")
    print(f"Sample 60-70: {test_filtered[60:70]}")
    print(f"Sample 500-510: {test_filtered[500:510]}")

    # The filter transient might be causing the energy imbalance

    print("\n5. ATTEMPTING ENERGY BALANCE FIX")
    print("-" * 40)

    # What if we normalize the preamble and payload separately?
    # This is just for testing - not suggesting we change the TX

    # Get the PN and data for frame 0
    pn_full = tx.sec.pn_bits(0, 1087)
    pn_payload = pn_full[63:]
    pn_symbols = 2.0 * pn_payload.astype(np.float32) - 1.0

    # Try despreading the original frame
    payload_original = frame[63:]
    despread_original = payload_original * pn_symbols

    print(f"Original despread std: {np.std(despread_original):.3f}")

    # Now let's check if the issue is the filter's frequency response
    print("\n6. FILTER FREQUENCY RESPONSE CHECK")
    print("-" * 40)

    # The bandpass filter might be attenuating our signal too much
    # or introducing phase distortion

    from scipy.signal import freqz
    w, h = freqz(b, a, fs=48000)

    # Find the passband
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    passband_mask = mag_db > -3  # Within 3dB of peak

    if np.any(passband_mask):
        passband_freqs = w[passband_mask]
        print(f"Passband: {passband_freqs[0]:.0f} - {passband_freqs[-1]:.0f} Hz")
        print(f"Designed band: {band[0]} - {band[1]} Hz")

    # Check phase response in passband
    phase = np.angle(h)
    if np.any(passband_mask):
        phase_variation = np.std(phase[passband_mask])
        print(f"Phase variation in passband: {phase_variation:.3f} radians")

    print("\n7. CONCLUSION")
    print("-" * 40)

    # Based on all the analysis, the issue is likely one of:
    # 1. Filter transient affecting energy distribution
    # 2. Energy imbalance between preamble and payload
    # 3. Phase distortion from the filter

    # Let's check if it's working at all by trying many frame counters
    print("Trying multiple frame counters...")

    for ctr in range(10):
        llr = rx._llr(frame, ctr)
        blob = polar_dec(llr)
        if blob is not None:
            try:
                plain = rx.sec.open(blob)
                if plain.startswith(b"ESAL"):
                    embedded_ctr = int.from_bytes(plain[4:8], 'big')
                    print(f"✅ SUCCESS with ctr={ctr}, embedded_ctr={embedded_ctr}")
                    return True
            except:
                pass

    print("❌ Failed to decode with any frame counter 0-9")

    # The real issue might be simpler - let's check if butter_bandpass is working correctly
    print("\n8. BUTTERWORTH FILTER CHECK")
    print("-" * 40)
    print(f"Filter order: {len(b) - 1}")
    print(f"b coefficients: {b}")
    print(f"a coefficients: {a}")

    # A high-order Butterworth can have numerical issues
    # The default order is 4, but we're using order 8 (from len(b)-1 = 8)

    return False


if __name__ == "__main__":
    final_diagnosis()