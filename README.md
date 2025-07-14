# 🔊 RTWM — Real-Time Ultrasonic Audio Watermarking

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/your-org/rtwm/test.yml?branch=main)](https://github.com/your-org/rtwm/actions)
[![PyPI](https://img.shields.io/pypi/v/rtwm)](https://pypi.org/project/rtwm/)

> 🔐 Embed and detect **tamper-proof ultrasonic watermarks** in live speech — in real-time, using AES-GCM encryption and Polar codes.

---

## 🎯 Features

- 🎙️ **Live Watermarking** — Transmit ultrasonic signals in real-time from mic input
- 🔐 **AES-GCM Encrypted Payloads** — Authenticated, unforgeable metadata
- 📶 **Polar Code ECC** — Robust to noise, compression, and recording artifacts
- 🧪 **Tamper Detection** — Detect missing, modified, or filtered audio
- 📁 **Offline Verification** — Check recorded files for watermark authenticity
- 🖥️ **GUI Tools** — Tkinter-based transmitter and receiver interfaces

---

## 📸 Screenshots

<p float="left">
  <img src="https://raw.githubusercontent.com/your-org/rtwm/main/docs/screenshot_tx_gui.png" width="400">
  <img src="https://raw.githubusercontent.com/your-org/rtwm/main/docs/screenshot_rx_gui.png" width="400">
</p>

---

## 🚀 Quickstart

### ▶️ Transmit (Live Watermark)

```bash
python tx_app.py --key 00112233445566778899aabbccddeeff
```

Or launch the GUI:

```bash
python tx_gui.py
```

### ✅ Verify (Recorded File)

```bash
python rx_app.py --key 00112233445566778899aabbccddeeff path/to/audio.wav
```

Or use:

```bash
python rx_gui.py
```

---

## 🧪 Run Tests

```bash
# install test deps
pip install -e .[dev]

# run all tests
pytest

# or use nox (recommended)
pip install nox
nox -s tests
```

---

## 🧠 How It Works

### TX (Transmitter):
- Encrypts metadata (e.g. session ID + frame counter) using AES-GCM
- Encodes it with Polar codes
- Modulates it via BPSK spread-spectrum
- Band-limits to 18–22 kHz and plays it in real-time

### RX (Receiver):
- Filters 18–22 kHz band from audio
- Despreads using PN chips
- Majority-votes bits and Polar-decodes
- Authenticates payload via AES-GCM
- Returns `Authentic` or `Tampered`

---

## 🔧 Dependencies

- numpy, scipy
- sounddevice, soundfile
- cryptography
- (Optional) pytest, black, mypy, nox

---

## 📂 Project Structure

```
rtwm/
├── audioio.py        # Full-duplex audio loop
├── crypto.py         # AES-GCM wrapper
├── detector.py       # Offline watermark verifier
├── embedder.py       # Real-time watermark embedder
├── polar.py          # Tiny polar code encoder/decoder
├── utils.py          # Filtering + chip generation
tests/
├── test_crypto.py
├── test_roundtrip.py
└── ...
```

---

## ⚖️ License

MIT License © RTWM Team

---

## 🌍 Links

- 📘 [Docs & Wiki](https://github.com/your-org/rtwm/wiki)
- 🐛 [Bug Tracker](https://github.com/your-org/rtwm/issues)
- 💬 [Discussion](https://github.com/your-org/rtwm/discussions)

---

## ⭐️ Acknowledgements

This project draws inspiration from real-time signal processing, ECC theory, and secure watermarking systems. Contributions and forks welcome!
