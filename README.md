# EchoSeal

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Real-time audio watermarking for speech. Embed a cryptographic mark while recording; verify any clip later and get **authentic**, **tampered**, or **no watermark** — with time ranges when something was edited.

The mark sits in the 4–22 kHz band, mixed below speech level. Each ~0.8 s frame is self-contained, so a few seconds of audio is enough to check.

---

## Idea

```
  live speech  →  embedder (TX)  →  watermarked audio  →  file / broadcast
                                                          ↓
                                              detector (RX)  →  verdict + issues
```

**TX** mixes an inaudible BPSK carrier into the audio stream in real time.  
**RX** finds frames, decrypts them, then checks that counters, timing, and waveforms all agree.

---

## Quick start

Requires Python 3.12+ and a working microphone for live TX.

```bash
git clone <repo-url> && cd EchoSeal
pip install -e ".[dev]"

# 256-bit key (64 hex chars)
export KEY=$(openssl rand -hex 32)

# Watermark from the mic for 30 s (optional: --save out.wav)
echoseal-tx --key $KEY --seconds 30

# Verify a recording
echoseal-rx --key $KEY --audio recording.wav
```

Exit codes from `echoseal-rx`: `0` authentic · `1` tampered · `2` no watermark.

**Docker** (verify only needs a mounted file):

```bash
docker build -t echoseal .
docker run --rm -v "$PWD:/data" echoseal echoseal-rx --key $KEY --audio /data/recording.wav
```

---

## What the detector checks

| Verdict | Meaning |
|---------|---------|
| `authentic` | Frames decode under your key; timeline and waveforms are consistent |
| `tampered` | Mark is present but something does not add up (see issues printed) |
| `no-watermark` | Nothing decodable — no mark, wrong key, or too degraded |

Tampering signals include cuts and inserts, spliced sessions, replayed frames, same-length replacements (e.g. dubbed words), and sub-frame edits. A watermark forged with another key never passes decryption.

---

## Stack (short)

- ChaCha20-Poly1305 payload + HKDF sub-keys (AEAD, PN, band hop)
- Polar code 1024/448, SCL decode
- Frequency-hopping BPSK in one of four sub-bands per frame
- 63-chip MLS preamble, 48 kHz, ~0.81 s per frame

More detail: [`EchoSeal_algorithm_summary.txt`](EchoSeal_algorithm_summary.txt).

---

## Project layout

```
rtwm/           core library (embedder, detector, crypto, polar, frame format)
tx_app.py       live transmitter CLI  →  echoseal-tx
rx_app.py       offline verifier CLI  →  echoseal-rx
tests/          pytest (unit + e2e scenarios)
gui/            optional TX/RX GUIs
```

---

## Tests

```bash
pytest                          # full suite
pytest --ignore=tests/test_e2e.py   # unit / component only
pytest tests/test_e2e.py        # scenario tests (cut, insert, AI swap, …)
```

---

## Status

Early research prototype. Validated on synthetic speech and controlled edits in tests. **Not yet** benchmarked for real room capture, phone microphones, or codec chains (MP3/AAC). Silence carries no mark by design.

---

## License

MIT — see [LICENSE](LICENSE).
