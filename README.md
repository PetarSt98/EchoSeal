# EchoSeal
# RTWM – Real-Time Audio Watermarking
Real-time audio watermarking and anti-tampering system protects live events from tampering, clipping etc.
```bash
# install in editable mode
$ pip install -e .

# start live transmitter (uses default mic/spk)
$ python tx_app.py --key 00112233445566778899aabbccddeeff

# verify a recording
$ python rx_app.py --key 00112233445566778899aabbccddeeff recording.wav
