import numpy as np

def test_frozen_bit_consistency():
    key = b"\xAA" * 32

    from rtwm.embedder import WatermarkEmbedder
    from rtwm.detector import WatermarkDetector
    from rtwm.fastpolar import PolarCode

    # Create instances
    tx = WatermarkEmbedder(key)
    rx = WatermarkDetector(key)
    pc = PolarCode(1024, 448)

    # Check frozen bit arrays
    print("Encoder frozen bits (first 10):", np.where(pc.frozen)[0][:10])
    print("Encoder frozen bits (last 10):", np.where(pc.frozen)[0][-10:])
    print("Encoder info positions (first 10):", np.where(~pc.frozen)[0][:10])
    print("Encoder info positions (last 10):", np.where(~pc.frozen)[0][-10:])

    # The detector should use the same polar code
    # If it doesn't, that's your problem!


if __name__ == "__main__":
    test_frozen_bit_consistency()