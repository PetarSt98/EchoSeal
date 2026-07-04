#!/usr/bin/env python
"""
CLI entry-point for live transmitter.
"""
from __future__ import annotations
import argparse, sys
from rtwm.embedder import WatermarkEmbedder
from rtwm.audioio import AudioLoop
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Real-time watermark transmitter")
    p.add_argument("--key", required=True, help="256-bit hex key (64 hex chars) or path to keyfile")
    p.add_argument("--device", type=int, help="sounddevice index")
    p.add_argument("--seconds", type=float, default=30.0, help="run duration")
    p.add_argument("--save", nargs="?", const="tx_output.wav", help="Save 10s to WAV file (default: tx_output.wav)")

    return p.parse_args()

def load_key(path_or_hex: str) -> bytes:
    stripped = path_or_hex.strip()
    if len(stripped) == 64 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return bytes.fromhex(stripped)
    with open(stripped, "rb") as f:
        return f.read()

def main():
    args = parse_args()
    key = load_key(args.key)
    if len(key) != 32:
        raise SystemExit("❌  Key must be 256-bit (64 hex chars).")
    embedder = WatermarkEmbedder(key)

    loop = AudioLoop(process_fn=embedder.process, fs=48_000, device=args.device, save_path=args.save)
    loop.start()
    print("▶ live watermarking – speak into mic …", file=sys.stderr)
    try:
        import time
        time.sleep(args.seconds)
    finally:
        loop.stop()

if __name__ == "__main__":
    main()
