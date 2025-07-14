"""
RT-Watermark-TX – simple Tkinter GUI
• select AES key (hex or file)
• optional audio-device index
• Start / Stop transmitter
• live microphone VU-meter (dBFS)
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue, threading, time, os, sys
import numpy as np
from rtwm.embedder import WatermarkEmbedder
from rtwm.audioio import AudioLoop


# --------------------------------------------------------------------- helpers
def load_key(path_or_hex: str) -> bytes:
    """Return 16/24/32-byte AES key from hex string or file path."""
    if all(c in "0123456789abcdefABCDEF" for c in path_or_hex.strip()) and len(
        path_or_hex.strip()
    ) in (32, 48, 64):
        return bytes.fromhex(path_or_hex.strip())
    path = os.path.expanduser(path_or_hex)
    return open(path, "rb").read()


def rms_dbfs(x: np.ndarray) -> float:
    """dBFS level of float32 mono signal."""
    peak = np.sqrt(np.mean(x**2) + 1e-12)
    return 20 * np.log10(peak + 1e-12)


# --------------------------------------------------------------------- main gui
class TxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RT-Watermark TX")
        self.resizable(False, False)

        # GUI state
        self._running = False
        self._level_q: queue.Queue[float] = queue.Queue(maxsize=10)
        self._audio_loop: AudioLoop | None = None
        self._tx_thread: threading.Thread | None = None

        # ---------------- widgets
        frm = ttk.Frame(self, padding=12)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="AES-GCM Key (hex or file):").grid(sticky="w")
        self.key_var = tk.StringVar(value="00112233445566778899aabbccddeeff")
        ttk.Entry(frm, width=50, textvariable=self.key_var).grid(row=0, column=1, pady=2)

        ttk.Label(frm, text="Audio device index (optional):").grid(row=1, column=0, sticky="w")
        self.dev_var = tk.StringVar()
        ttk.Entry(frm, width=10, textvariable=self.dev_var).grid(row=1, column=1, sticky="w")

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=2, columnspan=2, pady=6)
        self.start_btn = ttk.Button(btn_frm, text="Start", command=self._start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(btn_frm, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        ttk.Label(frm, text="Mic level").grid(row=3, column=0, sticky="w")
        self.vu = ttk.Progressbar(frm, length=250, maximum=60)  # 0..60 dB
        self.vu.grid(row=3, column=1, pady=4, sticky="w")

        self.status = ttk.Label(frm, text="Idle")
        self.status.grid(row=4, columnspan=2, pady=(8, 0))

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_level)

    # ---------------- audio & watermark
    def _embedder_factory(self) -> WatermarkEmbedder:
        key = load_key(self.key_var.get())
        return WatermarkEmbedder(key)

    def _audio_process(self, chunk: np.ndarray) -> np.ndarray:
        """Callback used in the audio thread."""
        level = rms_dbfs(chunk)
        try:
            self._level_q.put_nowait(level)
        except queue.Full:
            pass
        return self._embed.process(chunk)

    # ---------------- control buttons
    def _start(self):
        try:
            self._embed = self._embedder_factory()
        except Exception as e:
            messagebox.showerror("Key error", f"Failed to load key: {e}")
            return

        device = int(self.dev_var.get()) if self.dev_var.get() else None
        self._audio_loop = AudioLoop(self._audio_process, device=device)
        self._audio_loop.start()

        self._running = True
        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.status["text"] = "Running – watermark live"

    def _stop(self):
        if self._audio_loop:
            self._audio_loop.stop()
            self._audio_loop = None
        self._running = False
        self.start_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        self.status["text"] = "Stopped"

    # ---------------- housekeeping
    def _poll_level(self):
        try:
            level = self._level_q.get_nowait()
            self.vu["value"] = min(max(level + 60, 0), 60)  # map [-60..0] dBFS
        except queue.Empty:
            pass
        self.after(100, self._poll_level)

    def _on_close(self):
        self._stop()
        self.destroy()


if __name__ == "__main__":
    TxGUI().mainloop()
