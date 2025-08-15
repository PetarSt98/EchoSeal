import numpy as np
import pytest
from scipy.signal import lfilter

from rtwm.embedder import WatermarkEmbedder, TxParams
from rtwm.utils import choose_band, butter_bandpass
from rtwm.crypto import SecureChannel
from rtwm.polar_fast import encode as polar_enc

EPS = 1e-12
STEADY_OFFSET = 16  # must match embedder steady-state energy window


@pytest.fixture
def key():
    return b"\xAA" * 32


@pytest.fixture
def params():
    # Use defaults from code (fs=48k, N/K from polar layer)
    return TxParams()


def _manual_frame_chips_exact(tx: WatermarkEmbedder, payload: bytes) -> np.ndarray:
    """
    Exact re-implementation of WatermarkEmbedder._make_frame_chips()
    using *the same* session state and configuration.
    Differences vs older test:
      - Uses MLS(63) preamble like the embedder (not params.preamble)
      - Uses zero-state IIR with the *same dtype* for zi
      - Normalizes on steady-state part chips[16:], not full frame
    """
    band = choose_band(tx.sec.master_key, tx.frame_ctr)

    # Polar encode to 1024 bits
    data_bits = polar_enc(payload, N=tx.p.N, K=tx.p.K)

    # Build PN for whole frame and slice payload part
    pre_bits = tx.mseq_63()
    frame_len = pre_bits.size + data_bits.size
    pn_full = tx.sec.pn_bits(tx.frame_ctr, frame_len)
    pn_payload = pn_full[pre_bits.size:]

    # Map to BPSK and spread payload
    pre_sy = 2.0 * pre_bits.astype(np.float32) - 1.0
    data_sy = 2.0 * data_bits.astype(np.float32) - 1.0
    pn_sy   = 2.0 * pn_payload.astype(np.float32) - 1.0
    symbols = np.concatenate((pre_sy, data_sy * pn_sy))

    # Filter with zero initial state, matching embedder dtype
    b, a = butter_bandpass(*band, tx.p.fs, order=4)
    zi_len = max(len(a), len(b)) - 1
    zi = np.zeros(zi_len, dtype=np.result_type(a, b, symbols))
    chips, _ = lfilter(b, a, symbols, zi=zi)

    # Normalize using steady-state energy (avoid initial transient)
    steady = chips[STEADY_OFFSET:] if chips.size > STEADY_OFFSET else chips
    energy = float(np.mean(steady ** 2))
    if energy > EPS:
        chips /= np.sqrt(energy)

    return chips.astype(np.float32)


def test_manual_matches_embedder_exact(key, params):
    """_make_frame_chips must match a manual reimplementation *bit-for-bit (float)* when using the same payload."""
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = 0  # force known frame

    # Freeze the payload so both paths use identical bits
    payload = tx._build_payload()
    tx._build_payload = lambda: payload

    emb_chips = tx._make_frame_chips()
    man_chips = _manual_frame_chips_exact(tx, payload)

    assert emb_chips.shape == man_chips.shape
    assert np.allclose(emb_chips, man_chips, atol=1e-6), "Embedder diverged from exact manual implementation"


def test_frame_counter_ownership(key, params):
    """_make_frame_chips() must NOT increment the counter; process() *does*."""
    tx = WatermarkEmbedder(key, params)
    start = tx.frame_ctr
    _ = tx._make_frame_chips()
    assert tx.frame_ctr == start  # no change inside _make_frame_chips

    samples = np.zeros(1000, dtype=np.float32)
    _ = tx.process(samples)
    assert tx.frame_ctr == start + 1  # increment happens in process()


def test_payload_sealed_length_and_ctr_roundtrip(key, params):
    """Payload must be 55 bytes and include the frame counter in plaintext (validated via decrypt)."""
    tx = WatermarkEmbedder(key, params)
    for ctr in [0, 1, 7, 255]:
        tx.frame_ctr = ctr
        blob = tx._build_payload()
        assert isinstance(blob, bytes) and len(blob) == 55
        plain = SecureChannel(key).open(blob)
        assert plain.startswith(b"ESAL")
        assert int.from_bytes(plain[4:8], "big") == ctr


def test_chips_statistics_steady_state(key, params):
    """Chips must be unit-RMS on the steady part (as normalized by the embedder)."""
    tx = WatermarkEmbedder(key, params)
    chips = tx._make_frame_chips()
    steady = chips[STEADY_OFFSET:] if chips.size > STEADY_OFFSET else chips
    mean = float(np.mean(steady))
    rms  = float(np.sqrt(np.mean(steady ** 2)))
    assert abs(mean) < 5e-3, f"Steady-state mean too large: {mean}"
    assert abs(rms - 1.0) < 5e-3, f"Steady-state RMS not ~1: {rms}"


def test_silence_gate_returns_input(key, params):
    """Very quiet input should be returned unchanged (silence gate)."""
    tx = WatermarkEmbedder(key, params)
    # amplitude below MIN_RMS_SILENCE used in embedder
    samples = np.full(480, 1e-6, dtype=np.float32)
    out = tx.process(samples.copy())
    assert np.allclose(out, samples, atol=1e-12), "Silence gate should return input unchanged"


def test_no_clipping_headroom(key, params):
    """Mixer must not clip: output peak <= MIX_HEADROOM."""
    tx = WatermarkEmbedder(key, params)
    # A hot block; watermark should be scaled down by limiter
    samples = np.full(4096, 0.97, dtype=np.float32)
    out = tx.process(samples.copy())
    assert float(np.max(np.abs(out))) <= 0.98001, "Output exceeded headroom limit"


def test_preamble_correlation_has_dominant_peak(key, params):
    """
    Filtered preamble should produce a strong correlation right at the frame start.
    We use normalized cross-correlation (cosine similarity) at lag 0 and
    require it to be a clear statistical outlier vs. other lags.
    """
    tx = WatermarkEmbedder(key, params)
    tx.frame_ctr = 3
    chips = tx._make_frame_chips()

    # Build filtered preamble template in the same band
    band = choose_band(tx.sec.master_key, tx.frame_ctr)
    b, a = butter_bandpass(*band, tx.p.fs, order=4)
    pre_bits = tx.mseq_63()
    pre_sy = 2.0 * pre_bits.astype(np.float32) - 1.0

    zi_len = max(len(a), len(b)) - 1
    zi = np.zeros(zi_len, dtype=np.result_type(a, b, pre_sy))
    tpl, _ = lfilter(b, a, pre_sy, zi=zi)
    L = tpl.size

    # --- NCC at start (lag 0) ---
    seg0 = chips[:L]
    denom0 = (np.linalg.norm(seg0) * np.linalg.norm(tpl)) + 1e-12
    ncc0 = float(np.dot(seg0, tpl) / denom0)

    # --- Sliding NCC over the rest of the frame (exclude a small start window) ---
    START_WIN = max(16, zi_len)            # ignore early transient
    ncc_vals = []
    for i in range(START_WIN, chips.size - L + 1):
        seg = chips[i:i+L]
        denom = (np.linalg.norm(seg) * np.linalg.norm(tpl)) + 1e-12
        ncc_vals.append(float(np.dot(seg, tpl) / denom))
    ncc_vals = np.array(ncc_vals, dtype=np.float64)
    if ncc_vals.size == 0:  # safety
        pytest.skip("Frame too short after transient cut")

    # --- Criteria ---
    # 1) Start NCC should be healthy on an absolute scale (no crazy small value)
    assert ncc0 > 0.35, f"Start NCC too low: {ncc0:.3f}"

    # 2) Start NCC should be a statistical outlier vs. others (z-score)
    mu = float(ncc_vals.mean())
    sigma = float(ncc_vals.std() + 1e-12)
    z = (ncc0 - mu) / sigma
    assert z > 2.0, f"Preamble start NCC not dominant enough (z={z:.2f}, ncc0={ncc0:.3f})"

    # 3) Also require it to exceed most other lags (95th percentile)
    q95 = float(np.quantile(ncc_vals, 0.95))
    assert ncc0 >= q95, f"Start NCC {ncc0:.3f} < 95th percentile {q95:.3f} of other lags"


def test_choose_band_is_deterministic(key, params):
    """choose_band must be a pure function of (key, frame_ctr)."""
    for ctr in [0, 1, 5, 17, 255]:
        assert choose_band(key, ctr) == choose_band(key, ctr)
