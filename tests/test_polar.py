import os, numpy as np
from rtwm.polar_fast import encode, decode, N_DEFAULT, K_DEFAULT

def test_polar_roundtrip():
    payload = os.urandom(K_DEFAULT // 8)          # 56 bytes
    code    = encode(payload)                     # defaults N=1024/K=448
    assert code.size == N_DEFAULT
    assert decode(code) == payload
