import numpy as np
from scipy.signal import freqz

from rtwm.utils import butter_bandpass


def test_bandpass_passband_and_stopband():
    fs, lo, hi = 48_000, 8_000, 10_000
    b, a = butter_bandpass(lo, hi, fs, order=4)
    w, h = freqz(b, a, worN=8192)
    freq = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)

    center = int(np.argmin(np.abs(freq - (lo + hi) / 2)))
    assert mag_db[center] > -3.0  # ~0 dB in the passband

    below = int(np.argmin(np.abs(freq - 2_000)))
    above = int(np.argmin(np.abs(freq - 16_000)))
    assert mag_db[below] < -30.0
    assert mag_db[above] < -30.0


def test_bandpass_is_stable():
    b, a = butter_bandpass(4_000, 6_000, 48_000, order=4)
    assert np.all(np.abs(np.roots(a)) < 1.0)  # poles inside the unit circle
