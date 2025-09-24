import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np


def _load_rtwm_module(module: str):
    root = Path(__file__).resolve().parents[1] / "rtwm"
    pkg = sys.modules.get("rtwm")
    if pkg is None:
        pkg = types.ModuleType("rtwm")
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules["rtwm"] = pkg

    name = f"rtwm.{module}"
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(name, root / f"{module}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


polar_fast = _load_rtwm_module("polar_fast")
fastpolar = _load_rtwm_module("fastpolar")

encode = polar_fast.encode
decode = polar_fast.decode
N_DEFAULT = polar_fast.N_DEFAULT
K_DEFAULT = polar_fast.K_DEFAULT
PolarCode = fastpolar.PolarCode

def test_polar_roundtrip():
    crc_size = 8
    info_bits = K_DEFAULT - crc_size  # 440 bits → 55 bytes
    payload = np.frombuffer(os.urandom(info_bits // 8), dtype="u1")
    bits = np.unpackbits(payload)

    assert bits.size == info_bits

    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8, crc_size=crc_size)
    encoded = pc.encode(bits)

    assert encoded.size == N_DEFAULT

    llr = np.where(encoded == 1, 10.0, -10.0).astype(np.float32)

    decoded_bits, ok = pc.decode(llr)
    assert decoded_bits.size == info_bits
    assert ok is True

    recovered = np.packbits(decoded_bits).tobytes()
    assert recovered == payload.tobytes()


def test_polar_awgn_roundtrip():
    """Polar encoder/decoder should round-trip through an AWGN channel."""

    rng = np.random.default_rng(1234)

    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8, crc_size=8)
    info_len = pc.K - pc.crc_size
    info_bits = rng.integers(0, 2, info_len, dtype=np.uint8)

    codeword = pc.encode(info_bits)
    assert codeword.shape == (pc.N,)

    sigma = 0.15
    tx = 2.0 * codeword.astype(np.float64) - 1.0
    noise = rng.normal(0.0, sigma, size=tx.size)
    rx = tx + noise
    llr = 2.0 * rx / (sigma**2)

    decoded_bits, ok = pc.decode(llr)

    assert ok is True
    assert decoded_bits.shape == (info_len,)
    np.testing.assert_array_equal(decoded_bits, info_bits)


def test_polar_fast_wrapper_awgn_roundtrip():
    """The polar_fast convenience wrappers should also survive AWGN."""

    rng = np.random.default_rng(4321)

    payload = rng.integers(0, 256, size=(K_DEFAULT - 8) // 8, dtype=np.uint8).tobytes()

    codeword = encode(payload)
    assert codeword.shape == (N_DEFAULT,)

    sigma = 0.15
    tx = 2.0 * codeword.astype(np.float64) - 1.0
    noise = rng.normal(0.0, sigma, size=tx.size)
    rx = tx + noise
    llr = 2.0 * rx / (sigma**2)

    recovered, ok = decode(llr, return_ok=True)

    assert ok is True
    assert recovered == payload

def test_crc8():
    pc = PolarCode(1024, 448)
    bits = np.random.randint(0, 2, 440, dtype=np.uint8)
    crc = pc._crc8(bits)
    assert np.all(pc._crc8(bits) == crc)
    assert len(crc) == 8


def test_polar_with_noise():
    pc = PolarCode(N_DEFAULT, K_DEFAULT, list_size=8, crc_size=8)
    bits = np.random.randint(0, 2, K_DEFAULT - 8, dtype=np.uint8)
    enc = pc.encode(bits)

    # Add some noise
    snr_db = 10  # Start with good SNR
    signal_power = 1.0
    noise_power = signal_power / (10 ** (snr_db / 10))

    # Convert to BPSK and add noise
    bpsk = 2 * enc.astype(float) - 1
    noisy = bpsk + np.random.normal(0, np.sqrt(noise_power), len(bpsk))

    # Calculate LLR
    llr = 2 * noisy / noise_power

    dec_bits, ok = pc.decode(llr)
    print(f"SNR: {snr_db}dB, Decode success: {ok}")
    if ok:
        print(f"BER: {np.mean(bits != dec_bits[:len(bits)]):.6f}")

