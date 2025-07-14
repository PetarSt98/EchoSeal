#!/usr/bin/env python
"""
CLI receiver – verify a WAV/FLAC/PCM file.
"""
from __future__ import annotations
import argparse, soundfile as sf
from rtwm.detector import WatermarkDetector

def parse_args():
    p = argparse.ArgumentParser(description="Verify watermark")
    p.add_argument("--key", required=True, help="hex key or key file")
    p.add_argument("audio", help="audio file to check")
    return p.parse_args()

def load_key(path_or_hex: str) -> bytes:
    if len(path_or_hex) in (32, 48, 64):
        return bytes.fromhex(path_or_hex)
    return open(path_or_hex, "rb").read()

def main():
    args = parse_args()
    key = load_key(args.key)
    data, fs = sf.read(args.audio, always_2d=False)
    if fs != 48_000:
        raise RuntimeError("please resample to 48 kHz for MVP")
    detector = WatermarkDetector(key)
    ok = detector.verify(data)
    print("✅  authentic" if ok else "⚠️  tampered / no watermark")

if __name__ == "__main__":
    main()
