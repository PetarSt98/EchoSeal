import os, numpy as np
from rtwm.polar_fast import encode, decode, N_DEFAULT, K_DEFAULT
from rtwm.fastpolar import PolarCode

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

def test_crc8():
    pc = PolarCode(1024, 448)
    bits = np.random.randint(0, 2, 440, dtype=np.uint8)
    crc = pc._crc8(bits)
    assert np.all(pc._crc8(bits) == crc)
    assert len(crc) == 8

