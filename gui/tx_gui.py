"""
EchoSeal-TX – real-time frequency-hopping watermark transmitter (GUI)
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue, threading, time, os
import numpy as np

from rtwm.embedder import WatermarkEmbedder
from rtwm.audioio  import AudioLoop


# ───────────────────── helpers ────────────────────────────────────────────
HEX_CHARS = "0123456789abcdefABCDEF"

def load_key(src: str) -> bytes:
    s = src.strip()
    if len(s) == 64 and all(c in HEX_CHARS for c in s):
        return bytes.fromhex(s)
    path = os.path.expanduser(s)
    return open(path, "rb").read()


def rms_dbfs(x: np.ndarray) -> float:
    peak = np.sqrt(np.mean(x ** 2) + 1e-12)
    return 20 * np.log10(peak + 1e-12)


# ───────────────────── GUI ────────────────────────────────────────────────
class TxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EchoSeal – Transmitter")
        self.resizable(False, False)
        self.style = ttk.Style(self); self.style.theme_use("clam")

        # runtime state
        self._audio_loop: AudioLoop | None = None
        self._level_q: queue.Queue[float] = queue.Queue(maxsize=20)

        # ─── widgets
        frm = ttk.Frame(self, padding=16)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="XChaCha20 key (64 hex) or file:").grid(sticky="w")
        self.key_var = tk.StringVar(value="0"*64)
        ttk.Entry(frm, width=60, textvariable=self.key_var).grid(row=0, column=1, pady=4)

        ttk.Label(frm, text="Audio-device index (optional):").grid(row=1, column=0, sticky="w")
        self.dev_var = tk.StringVar()
        ttk.Entry(frm, width=8, textvariable=self.dev_var).grid(row=1, column=1, sticky="w", pady=4)

        # buttons
        btn_box = ttk.Frame(frm); btn_box.grid(row=2, columnspan=2, pady=10)
        self.start_btn = ttk.Button(btn_box, text="Start", command=self._start)
        self.stop_btn  = ttk.Button(btn_box, text="Stop",  command=self._stop, state="disabled")
        self.start_btn.pack(side="left", padx=5); self.stop_btn.pack(side="left", padx=5)

        ttk.Label(frm, text="Mic level").grid(row=3, column=0, sticky="w")
        self.vu = ttk.Progressbar(frm, length=260, maximum=60)  # -60 dB → 0 dB
        self.vu.grid(row=3, column=1, sticky="w")

        self.status = ttk.Label(frm, text="Idle", font=("Helvetica", 11, "bold"))
        self.status.grid(row=4, columnspan=2, pady=(12,0))

        self.after(100, self._poll_vu)
        self._centre()

    # ─── callbacks
    def _start(self) -> None:
        try:
            key = load_key(self.key_var.get())
            if len(key) != 32:
                raise ValueError("Need 64-hex chars / 32-byte key")
            self._embed = WatermarkEmbedder(key)
        except Exception as e:
            messagebox.showerror("Key error", f"{e}")
            return

        device = int(self.dev_var.get()) if self.dev_var.get() else None
        self._audio_loop = AudioLoop(self._process, device=device)
        self._audio_loop.start()

        self.start_btn["state"] = "disabled"; self.stop_btn["state"] = "normal"
        self.status.config(text="Running – watermark live", foreground="green")

    def _stop(self) -> None:
        if self._audio_loop:
            self._audio_loop.stop(); self._audio_loop = None
        self.start_btn["state"] = "normal"; self.stop_btn["state"] = "disabled"
        self.status.config(text="Stopped", foreground="black")

    # audio callback
    def _process(self, chunk: np.ndarray) -> np.ndarray:
        try: self._level_q.put_nowait(rms_dbfs(chunk))
        except queue.Full: pass
        return self._embed.process(chunk)

    def _poll_vu(self) -> None:
        try:
            lvl = self._level_q.get_nowait()
            self.vu["value"] = min(max(lvl + 60, 0), 60)
        except queue.Empty:
            pass
        self.after(100, self._poll_vu)

    # helpers
    def _centre(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()  - w)//2
        y = (self.winfo_screenheight() - h)//2
        self.geometry(f"+{x}+{y}")

    def destroy(self) -> None:          # graceful shutdown on WM close
        self._stop()
        super().destroy()


if __name__ == "__main__":
    TxGUI().mainloop()
