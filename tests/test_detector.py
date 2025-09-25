import numpy as np
from scipy.signal   import lfilter
from rtwm.embedder  import WatermarkEmbedder
from rtwm.detector  import WatermarkDetector
from rtwm.crypto    import SecureChannel
from rtwm.utils     import choose_band, butter_bandpass

# We want to feed the detector *exactly* what it expects.
FS   = 48_000
KEY  = b"\xAA"*32
SEC  = SecureChannel(KEY)
BAND = choose_band(KEY, 0)

# -----------------------------------------------------------------------------
def _generate_clean_frame():
    """Return one clean frame embedded into a longer dummy recording."""
    tx = WatermarkEmbedder(KEY)
    chips = tx._make_frame_chips()  # length = 1087

    # Embed into 3s dummy recording (zero-padded)
    fs = FS
    full_len = int(3.0 * fs)
    signal = np.zeros(full_len, dtype=np.float32)

    # insert in middle
    start = 0
    signal[start:start + chips.size] = chips
    return signal
                           # length = 1 087

# -----------------------------------------------------------------------------
def test_detector_on_perfect_frame():
    """Detector should pass on a pristine frame with no host audio."""
    frame = _generate_clean_frame()
    ok = WatermarkDetector(KEY).verify(frame, FS)
    assert ok, "Detector failed on pristine watermark frame"

# -----------------------------------------------------------------------------
def test_frame_alignment_search():
    """
    Shift the frame by every offset from −40 … +40 chips and see where the
    detector starts succeeding.  If it only passes, say, at +31, you know
    the correlation peak is 31 chips late.
    """
    frame = _generate_clean_frame()
    det   = WatermarkDetector(KEY)

    results = {}
    for off in range(-40, 41):
        padded = np.pad(frame, (max(0,  off), max(0, -off)), mode='constant')
        ok = det.verify(padded, FS)
        results[off] = ok
        det.session_nonce = None                # reset anti-replay

    print("offset → passed:", {k:v for k,v in results.items() if v})
    assert any(results.values()), "Detector never locks within ±40 chips"

# -----------------------------------------------------------------------------
def test_pn_sign_convention():
    """
    Manually despread the payload chips with RX's PN slice and check that the
    mean sign matches TX's mapping (+1 ↔ bit=1, −1 ↔ bit=0).
    """
    tx_frame   = _generate_clean_frame()
    pre        = 63
    payload_tx = tx_frame[pre:]

    pn_full    = SEC.pn_bits(0, tx_frame.size)
    pn_payl    = 2*pn_full[pre:] - 1            # ±1

    despread   = payload_tx * pn_payl
    mean_sign  = np.mean(despread)

    print("mean(despread) =", mean_sign)
    # For random data we expect mean ≈ 0.  If it's ≈ +1 or −1, the sign is off.
    assert abs(mean_sign) < 0.2, "PN despreading sign looks wrong"


def test_matched_filter_captures_tx_channel():
    """Detector matched-filter taps must include the TX filter response."""
    det = WatermarkDetector(KEY)
    band = choose_band(KEY, 0)
    b, a = butter_bandpass(*band, FS, order=4)

    # Reference cascade impulse: TX filter (b, a) applied twice.
    imp = np.zeros(512, dtype=np.float32)
    imp[0] = 1.0
    g = lfilter(b, a, imp)
    cascade = np.convolve(g, g, mode="full")

    e = cascade * cascade
    cumulative = np.cumsum(e)
    total = cumulative[-1]
    idx = np.searchsorted(cumulative, 0.999 * total)
    cascade = cascade[:idx + 1]

    h_expected = cascade[::-1]
    h_expected /= np.sqrt(np.sum(h_expected * h_expected))

    h_actual = det._matched_filter_taps(band)

    assert h_actual.shape == h_expected.shape
    np.testing.assert_allclose(h_actual, h_expected, rtol=1e-6, atol=1e-6)
