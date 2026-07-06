#!/usr/bin/env python
"""
OTA demo: play a watermarked background signal on speakers only (no microphone).

A quiet speech-shaped noise carrier is watermarked and played into the room.
You talk normally; the phone records the acoustic mix (carrier + your voice).
Verify the phone file with ``echoseal-rx``.

This does NOT use the mic and does NOT full-duplex loop — no echo of your voice.
For live mic watermarking use ``echoseal-tx`` instead.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter

from rtwm.embedder import WatermarkEmbedder

FS = 48_000
BLOCK = 1024


def load_key(path_or_hex: str) -> bytes:
    stripped = path_or_hex.strip()
    if len(stripped) == 64 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return bytes.fromhex(stripped)
    with open(stripped, "rb") as f:
        return f.read()


class SpeechShapedCarrier:
    """Low-pass noise host so the embedder silence gate stays open."""

    def __init__(self, *, fs: int = FS, rms: float = 0.08, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        b, a = butter(4, 3_000 / (fs / 2), "low")
        self._b, self._a = b, a
        self._zi = np.zeros(max(len(a), len(b)) - 1, dtype=np.float64)
        self._rms = rms

    def next_block(self, n: int) -> np.ndarray:
        white = self._rng.standard_normal(n)
        y, self._zi = lfilter(self._b, self._a, white, zi=self._zi)
        y = y.astype(np.float32)
        level = float(np.sqrt(np.mean(y * y)) + 1e-12)
        return (y * (self._rms / level)).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play watermarked background audio on speakers (no mic).",
        epilog=(
            "OTA phone test:\n"
            "  1. Run this script — speakers play a quiet watermarked carrier.\n"
            "  2. Give your talk in the room (mic on laptop is not used).\n"
            "  3. Record with a phone 1–3 m away.\n"
            "  4. Verify: echoseal-rx --key KEY --audio phone.wav\n"
            "\n"
            "The phone file must contain enough of the played carrier to decode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("seconds", type=float, help="How long to play (seconds)")
    p.add_argument("--key", required=True, help="256-bit hex key (64 chars) or key file path")
    p.add_argument("--device", type=int, help="output device index (default: system default)")
    p.add_argument(
        "--level",
        type=float,
        default=0.08,
        help="carrier RMS level before watermark mix (default: 0.08, ~-22 dBFS)",
    )
    p.add_argument(
        "--save",
        nargs="?",
        const="ota_playback.wav",
        metavar="PATH",
        help="Save played audio to WAV (default: ota_playback.wav)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0:
        raise SystemExit("seconds must be positive")

    key = load_key(args.key)
    if len(key) != 32:
        raise SystemExit("Key must be 256-bit (64 hex chars or 32 raw bytes).")

    embedder = WatermarkEmbedder(key)
    carrier = SpeechShapedCarrier(rms=args.level)
    save_buf: list[np.ndarray] = []
    save_max = int(FS * args.seconds) if args.save else 0
    saved = 0

    def callback(outdata, frames, _time, status) -> None:
        nonlocal saved
        if status:
            print("⚠", status, flush=True)
        host = carrier.next_block(frames)
        out = embedder.process(host)
        if args.save and saved < save_max:
            take = out[: min(out.size, save_max - saved)]
            save_buf.append(take.copy())
            saved += take.size
        outdata[:, 0] = out

    print("EchoSeal OTA playback (speakers only, no mic)", file=sys.stderr)
    print(f"  Duration : {args.seconds:.1f} s", file=sys.stderr)
    print(f"  Carrier  : speech-shaped noise @ RMS {args.level:.3f}", file=sys.stderr)
    print("  Path     : synth carrier → watermark → speakers", file=sys.stderr)
    if args.save:
        print(f"  Save     : {args.save}", file=sys.stderr)
    print(file=sys.stderr)
    print("Talk in the room; phone records the mix. No laptop mic is used.", file=sys.stderr)

    stream = sd.OutputStream(
        samplerate=FS,
        channels=1,
        dtype="float32",
        blocksize=BLOCK,
        device=args.device,
        callback=callback,
    )

    stream.start()
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("\nStopped early.", file=sys.stderr)
    finally:
        stream.stop()
        stream.close()
        if args.save and save_buf:
            audio = np.concatenate(save_buf)
            sf.write(args.save, audio, FS)
            print(f"Saved {audio.size / FS:.1f}s to {args.save}", file=sys.stderr)
        print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
