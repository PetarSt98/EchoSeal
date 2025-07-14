def test_truncated_signal_fails():
    key = b"\xDD" * 16
    tx = WatermarkEmbedder(key)
    speech = (np.random.randn(FS * SECS) * 0.05).astype(np.float32)
    wm = tx.process(speech)
    partial = wm[:FS * 2]  # only 2 seconds
    rx = WatermarkDetector(key)
    try:
        rx.verify(partial)
    except ValueError:
        pass  # expected due to MIN_SEC=3
    else:
        assert False, "should raise on too-short input"
