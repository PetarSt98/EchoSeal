#!/usr/bin/env python
"""
CLI entry-point for live transmitter.
"""
from __future__ import annotations
import argparse, sys
from rtwm.embedder import WatermarkEmbedder
from rtwm.audioio import AudioLoop

def parse_args():
    p = argparse.ArgumentParser(description="Real-time watermark transmitter")
    p.add_argument("--key", required=True, help="256-bit hex key (64 hex chars) or path to keyfile")
    p.add_argument("--device", type=int, help="sounddevice index")
    p.add_argument("--seconds", type=float, default=30.0, help="run duration")
    return p.parse_args()

def load_key(path_or_hex: str) -> bytes:
    if len(path_or_hex) in (32, 48, 64):  # hex string
        return bytes.fromhex(path_or_hex)
    return open(path_or_hex, "rb").read()

def main():
    args = parse_args()
    key = load_key(args.key)
    if len(key) != 32:
        raise SystemExit("❌  Key must be 256-bit (64 hex chars).")
    embedder = WatermarkEmbedder(key)
    loop = AudioLoop(process_fn=embedder.process, fs=48_000, device=args.device)
    loop.start()
    print("▶ live watermarking – speak into mic …", file=sys.stderr)
    try:
        import time
        time.sleep(args.seconds)
    finally:
        loop.stop()

if __name__ == "__main__":
    main()
