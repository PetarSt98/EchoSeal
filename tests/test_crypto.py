import os, pytest
from rtwm.crypto import SecureChannel
from rtwm.embedder import WatermarkEmbedder
from rtwm.detector import WatermarkDetector
import numpy as np

FS, SECS = 48_000, 3

def test_encrypt_decrypt_cycle():
    key = os.urandom(32)
    sc  = SecureChannel(key)
    plaintext = b"Authenticated message 123"
    blob = sc.seal(plaintext)
    assert sc.open(blob) == plaintext

def test_invalid_tag_detection():
    key = os.urandom(32)
    sc  = SecureChannel(key)
    tampered = bytearray(sc.seal(b"data"))
    tampered[-1] ^= 0x55
    with pytest.raises(Exception):
        sc.open(tampered)

def test_wrong_key_fails():
    key_tx, key_rx = b"\x11"*32, b"\x22"*32
    speech = (np.random.randn(FS*SECS) * 0.05).astype(np.float32)
    wm  = WatermarkEmbedder(key_tx).process(speech)
    ok  = WatermarkDetector(key_rx).verify(wm, FS)
    assert ok is False
