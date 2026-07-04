import os

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from rtwm.crypto import SecureChannel


def test_seal_open_roundtrip():
    sc = SecureChannel(os.urandom(32))
    msg = b"Authenticated message 123"
    assert sc.open(sc.seal(msg)) == msg


def test_seal_output_length():
    sc = SecureChannel(os.urandom(32))
    # nonce(12) + ciphertext(27) + tag(16)
    assert len(sc.seal(b"x" * 27)) == 12 + 27 + 16


def test_nonce_is_random_per_seal():
    sc = SecureChannel(os.urandom(32))
    assert sc.seal(b"same") != sc.seal(b"same")


def test_tamper_is_detected():
    sc = SecureChannel(os.urandom(32))
    blob = bytearray(sc.seal(b"data"))
    blob[-1] ^= 0x55
    with pytest.raises(InvalidTag):
        sc.open(bytes(blob))


def test_wrong_key_cannot_open():
    a, b = SecureChannel(b"\x11" * 32), SecureChannel(b"\x22" * 32)
    with pytest.raises(InvalidTag):
        b.open(a.seal(b"secret"))


def test_open_rejects_short_blob():
    sc = SecureChannel(os.urandom(32))
    with pytest.raises(ValueError):
        sc.open(b"too short")


def test_requires_256bit_key():
    with pytest.raises(ValueError):
        SecureChannel(b"\x00" * 16)


def test_pn_bits_deterministic_and_keyed():
    key = os.urandom(32)
    bits = SecureChannel(key).pn_bits(7, 512)

    assert bits.shape == (512,)
    assert set(np.unique(bits)).issubset({0, 1})
    assert np.array_equal(bits, SecureChannel(key).pn_bits(7, 512))
    assert not np.array_equal(bits, SecureChannel(key).pn_bits(8, 512))
    assert not np.array_equal(bits, SecureChannel(os.urandom(32)).pn_bits(7, 512))


def test_band_key_derived_and_stable():
    key = os.urandom(32)
    assert len(SecureChannel(key).band_key) == 32
    assert SecureChannel(key).band_key == SecureChannel(key).band_key
    assert SecureChannel(key).band_key != SecureChannel(os.urandom(32)).band_key
