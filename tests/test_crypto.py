import os, pytest
from cryptography.exceptions import InvalidTag
from rtwm.crypto import AESCipher

FS   = 48_000
SECS = 3

def test_encrypt_decrypt_cycle():
    key = os.urandom(16)
    aes = AESCipher(key)
    plaintext = b"Authenticated message 123"
    blob = aes.encrypt(plaintext)
    assert aes.decrypt(blob) == plaintext

def test_invalid_tag_detection():
    key = os.urandom(16)
    aes = AESCipher(key)
    bad = bytearray(aes.encrypt(b"data"))
    bad[-1] ^= 0x55          # flip one bit
    with pytest.raises(InvalidTag):
        aes.decrypt(bad)

def test_wrong_key_fails():
    key_tx = b"\x11" * 16
    key_rx = b"\x22" * 16
    speech = (np.random.randn(FS * SECS) * 0.05).astype(np.float32)
    tx = WatermarkEmbedder(key_tx)
    wm = tx.process(speech)
    rx = WatermarkDetector(key_rx)
    assert rx.verify(wm) is False, "decryption with wrong key should fail"
