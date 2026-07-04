"""
EchoSeal v0.2 — live watermark transmitter (GUI).
"""
from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd

from gui.common import load_key, random_key_hex
from rtwm.audioio import AudioLoop
from rtwm.embedder import WatermarkEmbedder


def rms_dbfs(x: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(x**2) + 1e-12) + 1e-12)


def list_io_devices() -> list[tuple[str, str]]:
    """Return (label, device_index_or_empty) for full-duplex devices."""
    out: list[tuple[str, str]] = [("System default", "")]
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and dev["max_output_channels"] > 0:
                out.append((f"{i}: {dev['name']}", str(i)))
    except Exception:
        pass
    return out


class TxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EchoSeal v0.2 — Transmitter")
        self.minsize(520, 360)
        ttk.Style(self).theme_use("clam")

        self._audio_loop: AudioLoop | None = None
        self._embed: WatermarkEmbedder | None = None
        self._level_q: queue.Queue[float] = queue.Queue(maxsize=20)

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="Live speech watermarking (48 kHz, ChaCha20-Poly1305)",
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        # key
        ttk.Label(root, text="Key (64 hex chars or file):").grid(row=1, column=0, sticky="w")
        self.key_var = tk.StringVar()
        key_row = ttk.Frame(root)
        key_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Entry(key_row, width=48, textvariable=self.key_var).pack(side="left", fill="x", expand=True)
        ttk.Button(key_row, text="File…", command=self._pick_key, width=6).pack(side="left", padx=(4, 0))
        ttk.Button(key_row, text="Random", command=self._gen_key, width=7).pack(side="left", padx=(4, 0))

        # device
        ttk.Label(root, text="Audio device:").grid(row=2, column=0, sticky="w")
        self._devices = list_io_devices()
        self.dev_var = tk.StringVar(value=self._devices[0][1])
        self.dev_combo = ttk.Combobox(
            root,
            width=46,
            state="readonly",
            values=[d[0] for d in self._devices],
        )
        self.dev_combo.current(0)
        self.dev_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        # save
        save_frm = ttk.LabelFrame(root, text="Optional WAV capture", padding=8)
        save_frm.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.save_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(save_frm, text="Save watermarked output on stop", variable=self.save_var).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(save_frm, text="Path:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.save_path_var = tk.StringVar(value=os.path.abspath("tx_output.wav"))
        path_row = ttk.Frame(save_frm)
        path_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Entry(path_row, textvariable=self.save_path_var, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(path_row, text="…", command=self._pick_save, width=3).pack(side="left", padx=(4, 0))
        ttk.Label(save_frm, text="Seconds:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.save_secs_var = tk.StringVar(value="10")
        ttk.Spinbox(save_frm, from_=1, to=120, width=6, textvariable=self.save_secs_var).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )

        # controls
        btn_row = ttk.Frame(root)
        btn_row.grid(row=4, column=0, columnspan=3, pady=14)
        self.start_btn = ttk.Button(btn_row, text="Start", command=self._start)
        self.stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop, state="disabled")
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn.pack(side="left", padx=4)

        ttk.Label(root, text="Input level").grid(row=5, column=0, sticky="w")
        self.vu = ttk.Progressbar(root, length=320, maximum=60)
        self.vu.grid(row=5, column=1, columnspan=2, sticky="ew")

        self.status = ttk.Label(root, text="Idle", font=("Segoe UI", 11, "bold"))
        self.status.grid(row=6, column=0, columnspan=3, pady=(12, 0), sticky="w")

        root.columnconfigure(1, weight=1)
        self.after(100, self._poll_vu)
        self._centre()

    def _pick_key(self) -> None:
        path = filedialog.askopenfilename(title="Open key file", filetypes=[("Key", "*.*")])
        if path:
            self.key_var.set(path)

    def _gen_key(self) -> None:
        self.key_var.set(random_key_hex())

    def _pick_save(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save watermarked WAV",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
        )
        if path:
            self.save_path_var.set(path)

    def _device_index(self) -> int | None:
        idx = self.dev_combo.current()
        val = self._devices[idx][1]
        return int(val) if val else None

    def _start(self) -> None:
        try:
            key = load_key(self.key_var.get())
            if len(key) != 32:
                raise ValueError("Key must be 256 bits (64 hex characters or 32 raw bytes).")
            self._embed = WatermarkEmbedder(key)
        except Exception as exc:
            messagebox.showerror("Key error", str(exc))
            return

        save_path: str | None = None
        save_seconds = 10.0
        if self.save_var.get():
            save_path = self.save_path_var.get().strip()
            if not save_path:
                messagebox.showerror("Save path", "Choose a WAV file path.")
                return
            try:
                save_seconds = float(self.save_secs_var.get())
                if save_seconds <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Save duration", "Enter a positive number of seconds.")
                return

        self._audio_loop = AudioLoop(
            self._process,
            fs=48_000,
            device=self._device_index(),
            save_path=save_path,
            save_seconds=save_seconds,
        )
        self._audio_loop.start()

        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.status.config(
            text="Running — watermark live on mic loopback",
            foreground="#1a7f37",
        )

    def _stop(self) -> None:
        saved = self._audio_loop.save_path if self._audio_loop else None
        if self._audio_loop:
            self._audio_loop.stop()
            self._audio_loop = None
        self.start_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        if saved:
            self.status.config(text=f"Stopped — saved to {saved}", foreground="#0969da")
        else:
            self.status.config(text="Stopped", foreground="black")

    def _process(self, chunk: np.ndarray) -> np.ndarray:
        assert self._embed is not None
        try:
            self._level_q.put_nowait(rms_dbfs(chunk))
        except queue.Full:
            pass
        return self._embed.process(chunk)

    def _poll_vu(self) -> None:
        try:
            lvl = self._level_q.get_nowait()
            self.vu["value"] = min(max(lvl + 60, 0), 60)
        except queue.Empty:
            pass
        self.after(100, self._poll_vu)

    def _centre(self) -> None:
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")

    def destroy(self) -> None:
        self._stop()
        super().destroy()


def main() -> None:
    TxGUI().mainloop()


if __name__ == "__main__":
    main()
