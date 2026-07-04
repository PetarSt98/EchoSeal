#!/usr/bin/env python
"""
CLI receiver – verify a recording (WAV/FLAC/OGG …).

Exit codes: 0 = authentic, 1 = tampered, 2 = no watermark found.
"""
from __future__ import annotations

import argparse

import soundfile as sf

from rtwm.detector import WatermarkDetector


def parse_args():
    p = argparse.ArgumentParser(description="Verify EchoSeal watermark")
    p.add_argument("--key", required=True, help="256-bit hex key (64 hex chars) or path to keyfile")
    p.add_argument("--audio", required=True, help="audio file to check")
    return p.parse_args()


def load_key(path_or_hex: str) -> bytes:
    stripped = path_or_hex.strip()
    if len(stripped) == 64 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return bytes.fromhex(stripped)
    with open(stripped, "rb") as f:
        return f.read()


def main() -> None:
    args = parse_args()
    key = load_key(args.key)
    if len(key) != 32:
        raise SystemExit("Key must be 256-bit (64 hex chars).")

    data, fs = sf.read(args.audio, always_2d=False)
    report = WatermarkDetector(key).analyze(data, fs)

    if report.hits:
        first, last = report.hits[0], report.hits[-1]
        print(
            f"frames: {len(report.hits)} decoded "
            f"(counters {first.ctr}..{last.ctr}, "
            f"t={first.start / 48_000:.2f}s..{last.start / 48_000:.2f}s, "
            f"coverage {report.coverage:.0%})"
        )
    for issue in report.issues:
        print(f"issue: {issue}")
    if report.verdict == "authentic":
        for t0, t1 in report.unverified_spans:
            print(f"note: no verifiable watermark t={t0:.2f}s..{t1:.2f}s "
                  "(channel too degraded there to judge)")

    verdict = report.verdict
    print(
        {
            "authentic": "AUTHENTIC - watermark verified, timeline consistent",
            "tampered": "TAMPERED - watermark present but inconsistent",
            "no-watermark": "NO WATERMARK - nothing decodable (wrong key, no mark, or too degraded)",
        }[verdict]
    )
    raise SystemExit({"authentic": 0, "tampered": 1, "no-watermark": 2}[verdict])


if __name__ == "__main__":
    main()
