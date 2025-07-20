"""
sounddevice wrapper – full duplex, thread-safe stop, blocksize param.
"""

from __future__ import annotations
import sounddevice as sd
import numpy as np
from typing import Callable
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
    ) -> None:
        self.process = process_fn
        self.fs      = fs
        self.device  = device
        self.block   = block
        self._stream: sd.Stream | None = None
        self.save_path = save_path
        self._input_buffer = []
        self._output_buffer = []
        self._samples_to_save = fs * 10 if save_path else 0

    # --------------------------------------------------------------------- run
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

    # --------------------------------------------------------------------- callback
    def _callback(self, indata, outdata, frames, _time, status):
        if status:
            print("⚠", status, flush=True)
        input_signal = indata[:, 0]
        output_signal = self.process(input_signal)

        if self._samples_to_save > 0:
            self._input_buffer.append(input_signal.copy())
            self._output_buffer.append(output_signal.copy())
            self._samples_to_save -= len(input_signal)

        outdata[:] = output_signal.reshape(-1, 1)

    def _maybe_save(self):
        if self.save_path and self._output_buffer:
            output_audio = np.concatenate(self._output_buffer)[:self.fs * 10]
            sf.write(self.save_path, output_audio, self.fs)
            print(f"💾 Saved 10s sample to: {self.save_path}", flush=True)