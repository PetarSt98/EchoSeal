#!/usr/bin/env python3
"""
Test detection when recording starts at arbitrary points in the watermarked stream.
This simulates someone starting to record in the middle of a speech.
"""

import numpy as np
from scipy.signal import chirp
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector


def test_mid_stream_detection():
    """Test detection when recording starts at various points."""
    print("=" * 60)
    print("MID-STREAM DETECTION TEST")
    print("=" * 60)

    key = b"\xAA" * 32
    fs = 48000

    # Create a long watermarked signal (30 seconds)
    print("\n1. Creating 30-second watermarked signal...")
    duration = 30
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    speech = 0.3 * chirp(t, f0=300, f1=3500, t1=duration, method='linear').astype(np.float32)

    tx = WatermarkEmbedder(key)
    watermarked = tx.process(speech)

    total_frames = tx.frame_ctr
    print(f"   Total frames embedded: {total_frames}")
    print(f"   Approximate frame duration: {1087 / fs:.3f} seconds")

    # Test detection from various starting points
    test_points = [
        ("Beginning", 0),
        ("After 5 seconds", 5 * fs),
        ("After 10 seconds", 10 * fs),
        ("After 15 seconds", 15 * fs),
        ("After 20 seconds", 20 * fs),
        ("Middle of frame 50", int(50.5 * 1087)),
        ("Middle of frame 100", int(100.5 * 1087)),
        ("Random point 1", 123456),
        ("Random point 2", 654321),
        ("Near end", len(watermarked) - 5 * fs),
    ]

    print("\n2. Testing detection from various starting points:")
    print("-" * 60)

    results = []
    for name, start_pos in test_points:
        if start_pos >= len(watermarked) - 3 * fs:
            continue  # Skip if too close to end

        # Extract a 3-second clip starting from this position
        end_pos = min(start_pos + 3 * fs, len(watermarked))
        clip = watermarked[start_pos:end_pos]

        # Estimate which frame this should be
        estimated_frame = start_pos // 1087

        # Try detection
        rx = WatermarkDetector(key)
        result = rx.verify(clip, fs)

        status = "✓" if result else "✗"
        print(f"   {status} {name:20s} (pos {start_pos:7d}, ~frame {estimated_frame:3d}): {result}")
        results.append(result)

    success_rate = sum(results) / len(results)
    print(f"\n3. Success rate: {success_rate:.1%} ({sum(results)}/{len(results)})")

    return success_rate > 0.8


def test_continuous_stream_simulation():
    """Simulate continuous streaming with random start/stop recording."""
    print("\n" + "=" * 60)
    print("CONTINUOUS STREAM SIMULATION")
    print("=" * 60)

    key = b"\xAA" * 32
    fs = 48000

    # Create a long speech (60 seconds)
    print("\n1. Creating 60-second continuous stream...")
    duration = 60
    t = np.linspace(0, duration, fs * duration, endpoint=False)

    # More realistic speech simulation with varying amplitude
    speech = np.zeros_like(t)
    for i in range(10):
        start = i * 6
        end = start + 5
        mask = (t >= start) & (t < end)
        amp = 0.2 + 0.1 * np.random.rand()
        speech[mask] = amp * np.sin(2 * np.pi * (500 + 200 * i) * t[mask])

    speech = speech.astype(np.float32)

    # Embed watermark
    tx = WatermarkEmbedder(key)
    watermarked = tx.process(speech)

    print(f"   Embedded {tx.frame_ctr} frames")

    # Simulate multiple random recordings
    print("\n2. Simulating random recordings...")

    np.random.seed(42)
    num_recordings = 20
    results = []

    for i in range(num_recordings):
        # Random start time between 5 and 50 seconds
        start_time = np.random.uniform(5, 50)
        # Random duration between 3 and 10 seconds
        rec_duration = np.random.uniform(3, 10)

        start_sample = int(start_time * fs)
        end_sample = min(start_sample + int(rec_duration * fs), len(watermarked))

        recording = watermarked[start_sample:end_sample]

        # Try detection
        rx = WatermarkDetector(key)
        result = rx.verify(recording, fs)
        results.append(result)

        status = "✓" if result else "✗"
        print(
            f"   {status} Recording {i + 1:2d}: {start_time:5.1f}s - {start_time + rec_duration:5.1f}s ({rec_duration:.1f}s)")

    success_rate = sum(results) / len(results)
    print(f"\n3. Success rate: {success_rate:.1%} ({sum(results)}/{len(results)})")

    return success_rate > 0.8


def test_frame_counter_range():
    """Test detection with high frame counters (long streams)."""
    print("\n" + "=" * 60)
    print("HIGH FRAME COUNTER TEST")
    print("=" * 60)

    key = b"\xAA" * 32
    fs = 48000

    # Create embedder and set high frame counter (simulating long stream)
    tx = WatermarkEmbedder(key)

    # Test different starting frame counters
    test_counters = [0, 100, 500, 1000, 2000]
    results = []

    for start_ctr in test_counters:
        tx.frame_ctr = start_ctr

        # Generate 5 seconds
        duration = 5
        t = np.linspace(0, duration, fs * duration, endpoint=False)
        carrier = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        watermarked = tx.process(carrier)

        # Try detection
        rx = WatermarkDetector(key)
        result = rx.verify(watermarked, fs)
        results.append(result)

        status = "✓" if result else "✗"
        print(f"   {status} Starting from frame {start_ctr}: {result}")

    return all(results)


def test_robustness_to_clipping():
    """Test detection when recording is clipped at boundaries."""
    print("\n" + "=" * 60)
    print("CLIPPING ROBUSTNESS TEST")
    print("=" * 60)

    key = b"\xAA" * 32
    fs = 48000

    # Create short watermarked signal
    duration = 10
    t = np.linspace(0, duration, fs * duration, endpoint=False)
    carrier = 0.3 * chirp(t, f0=500, f1=2000, t1=duration, method='linear').astype(np.float32)

    tx = WatermarkEmbedder(key)
    watermarked = tx.process(carrier)

    # Test with various clip sizes
    clip_tests = [
        ("Full signal", 0, len(watermarked)),
        ("Missing first 0.5s", int(0.5 * fs), len(watermarked)),
        ("Missing last 0.5s", 0, len(watermarked) - int(0.5 * fs)),
        ("Missing both ends", int(0.5 * fs), len(watermarked) - int(0.5 * fs)),
        ("Only middle 3s", int(3.5 * fs), int(6.5 * fs)),
        ("Arbitrary clip", 12345, 12345 + 4 * fs),
    ]

    print("\nTesting with various clipping scenarios:")
    results = []

    for name, start, end in clip_tests:
        if start >= end or end > len(watermarked):
            continue

        clip = watermarked[start:end]
        clip_duration = len(clip) / fs

        rx = WatermarkDetector(key)
        result = rx.verify(clip, fs)
        results.append(result)

        status = "✓" if result else "✗"
        print(f"   {status} {name:20s} ({clip_duration:.1f}s): {result}")

    return sum(results) >= len(results) - 1  # Allow one failure


if __name__ == "__main__":
    print("Testing mid-stream detection capabilities...\n")

    test1 = test_mid_stream_detection()
    test2 = test_continuous_stream_simulation()
    test3 = test_frame_counter_range()
    test4 = test_robustness_to_clipping()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Mid-stream detection: {'PASS' if test1 else 'FAIL'}")
    print(f"Continuous stream: {'PASS' if test2 else 'FAIL'}")
    print(f"High frame counters: {'PASS' if test3 else 'FAIL'}")
    print(f"Clipping robustness: {'PASS' if test4 else 'FAIL'}")

    overall = test1 and test2 and test3 and test4
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")

    if not overall:
        print("\nNote: The detector may need tuning for better mid-stream detection.")
        print("Consider increasing MAX_SEARCH_CTR in detector._scan_band_multi_frame()")