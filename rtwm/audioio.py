"""
sounddevice wrapper – full duplex, thread-safe stop, blocksize param.
"""

from __future__ import annotations
import sounddevice as sd
import numpy as np
from typing import Callable

class AudioLoop:
    def __init__(
        self,
        process_fn: Callable[[np.ndarray], np.ndarray],
        *,
        fs: int = 48_000,
        device: int | str | None = None,
        block: int = 1_024,
    ) -> None:
        self.process = process_fn
        self.fs      = fs
        self.device  = device
        self.block   = block
        self._stream: sd.Stream | None = None

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

    # --------------------------------------------------------------------- callback
    def _callback(self, indata, outdata, frames, _time, status):
        if status:
            print("⚠", status, flush=True)
        outdata[:] = self.process(indata[:, 0]).reshape(-1, 1)
