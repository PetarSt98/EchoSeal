import numpy as np
from rtwm.embedder  import WatermarkEmbedder
from rtwm.detector  import WatermarkDetector
from rtwm.crypto    import SecureChannel
from rtwm.utils     import choose_band

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


def test_llr_alignment_handles_large_filter_delay():
    """The matched-filter shift search must cover long IIR tails (>64 chips)."""

    import types
    from rtwm.detector import FRAME_LEN, PRE_L, HDR_L, N_DEFAULT

    tx = WatermarkEmbedder(KEY)

    # Fix the payload so the generated frames are deterministic.
    fixed_payload = bytes(range(55))

    def _fixed_payload(self):
        return fixed_payload

    tx._build_payload = types.MethodType(_fixed_payload, tx)

    det = WatermarkDetector(KEY)

    # Generate a few consecutive frames; at least one hop band exhibits
    # group delay that requires a >64-chip correction when only TX filtering
    # is applied (mirrors the failing field capture).
    target = None
    best_s = None
    for ctr in range(8):
        frame = tx._make_frame_chips()[:FRAME_LEN].astype(np.float32)

        # Manual LLR reconstruction with an unrestricted shift search
        # (logic mirrored from WatermarkDetector._llr, but without the old cap).
        band = choose_band(KEY, ctr)
        h = det._matched_filter_taps(band)

        pn_full = det.sec.pn_bits(ctr, FRAME_LEN)
        pn_payload = pn_full[PRE_L + HDR_L :]
        pn_sy = 2.0 * pn_payload.astype(np.float32) - 1.0

        rx_pay = frame[PRE_L + HDR_L :].astype(np.float32)
        n = min(rx_pay.size, pn_sy.size)
        if n == 0:
            continue
        rx_pay = rx_pay[:n]
        pn_sy = pn_sy[:n]

        mf = np.convolve(rx_pay, h, mode="full").astype(np.float32)
        offset = len(h) - 1
        max_search = max(4 * len(h), HDR_L)
        margin = min(n // 2, max_search)
        start = max(0, offset - margin)
        stop = min(mf.size, offset + n + margin)
        mf_win = mf[start:stop]
        base = offset - start
        guard = int(max(16, min(64, len(h) // 8)))

        best_score = -1.0
        s_best = 0
        for s in range(-margin, margin + 1):
            i0 = base + s
            i1 = i0 + n
            if i0 < 0 or i1 > mf_win.size:
                continue
            aligned = mf_win[i0:i1]
            prod = aligned * pn_sy
            tail = prod[guard:] if prod.size > guard + 8 else prod
            score = float(np.mean(np.abs(tail)))
            if score > best_score:
                best_score = score
                s_best = s

        if abs(s_best) <= 64:
            continue  # look for a hop that actually needed the wider window

        i0 = base + s_best
        aligned = mf_win[i0 : i0 + n]
        prod = aligned * pn_sy
        tail = prod[guard:] if prod.size > guard + 8 else prod
        mu = float(np.mean(tail))
        llr_raw = prod - mu
        mad = float(np.median(np.abs(tail - float(np.median(tail))))) + 1e-12
        sigma_mad = 1.4826 * mad
        sigma_std = float(np.std(tail)) + 1e-12
        sigma = max(sigma_mad, sigma_std, 0.1)
        scale = float(np.clip(2.0 / (sigma * sigma), 0.5, 30.0))
        llr_manual = np.clip(llr_raw * scale, -12.0, 12.0).astype(np.float32)

        if llr_manual.size != N_DEFAULT:
            padded = np.zeros(N_DEFAULT, dtype=np.float32)
            m = min(llr_manual.size, N_DEFAULT)
            padded[:m] = llr_manual[:m]
            llr_manual = padded

        target = frame
        best_s = s_best
        manual_llr = llr_manual
        ctr_target = ctr
        break

    assert target is not None, "No frame exhibited >64-chip alignment error"

    llr_detector = det._llr(target, ctr_target)
    diff = np.linalg.norm(llr_detector - manual_llr)

    assert diff < 1e-2, (
        f"Matched-filter alignment failed to reach best shift {best_s}; "
        f"LLR mismatch norm={diff:.2f}"
    )
