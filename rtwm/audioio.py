"""
Thin full-duplex sounddevice wrapper for real-time processing.

The `process_fn` is called for every input block and must return an equally
sized output block that is written to the speakers.  Optionally the first
`save_seconds` of output are captured and written to `save_path` on stop.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioLoop:
    def __init__(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        *,
        fs: int = 48_000,
        device: int | str | None = None,
        block: int = 1_024,
        save_path: str | None = None,
        save_seconds: float = 10.0,
    ) -> None:
        self.process = process_fn
        self.fs = fs
        self.device = device
        self.block = block
        self.save_path = save_path
        self._stream: sd.Stream | None = None
        self._output_buffer: list[np.ndarray] = []
        self._save_max_samples = int(fs * save_seconds) if save_path else 0
        self._samples_to_save = self._save_max_samples

    # ----------------------------------------------------------------- run
    def start(self) -> None:
        if self._stream:
            return
        self._stream = sd.Stream(
            samplerate=self.fs,
            channels=1,
            blocksize=self.block,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
        self._maybe_save()

    # ------------------------------------------------------------- callback
    def _callback(self, indata, outdata, frames, _time, status) -> None:
        if status:
            print("⚠", status, flush=True)

        output = self.process(indata[:, 0])

        if self._samples_to_save > 0:
            self._output_buffer.append(output.copy())
            self._samples_to_save -= output.size

        outdata[:] = output.reshape(-1, 1)

    def _maybe_save(self) -> None:
        if not (self.save_path and self._output_buffer):
            return
        audio = np.concatenate(self._output_buffer)
        if self._save_max_samples:
            audio = audio[: self._save_max_samples]
        sf.write(self.save_path, audio, self.fs)
        print(f"Saved {audio.size / self.fs:.1f}s to: {self.save_path}", flush=True)
