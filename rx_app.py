#!/usr/bin/env python
"""
CLI receiver – verify a WAV/FLAC/PCM file.
"""
from __future__ import annotations
import argparse, soundfile as sf
from rtwm.detector import WatermarkDetector

def parse_args():
    p = argparse.ArgumentParser(description="Verify watermark")
    p.add_argument("--key", required=True, help="256-bit hex key (64 hex chars) or path to keyfile")
    p.add_argument("audio", help="audio file to check")
    return p.parse_args()

def load_key(path_or_hex: str) -> bytes:
    stripped = path_or_hex.strip()
    if len(stripped) in (32, 48, 64) and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return bytes.fromhex(stripped)
    return open(stripped, "rb").read()

def main():
    args = parse_args()
    key = load_key(args.key)
    if len(key) != 32:
        raise SystemExit("❌  Key must be 256-bit (64 hex chars).")
    data, fs = sf.read(args.audio, always_2d=False)
    detector = WatermarkDetector(key)
    ok = detector.verify(data, fs)
    print("✅  authentic" if ok else "⚠️  tampered / no watermark")

if __name__ == "__main__":
    main()
