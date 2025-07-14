import os, numpy as np
from rtwm.polar import polar_encode, polar_decode

N, K = 512, 344   # match TxParams defaults

def test_encode_decode_roundtrip():
    payload = os.urandom(K // 8)     # 43 bytes
    code = polar_encode(payload, N=N, K=K)
    assert code.size == N
    decoded = polar_decode(code, N=N, K=K)
    assert decoded == payload
