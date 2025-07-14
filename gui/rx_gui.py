"""
EchoSeal-RX – offline verifier GUI
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, soundfile as sf

from rtwm.detector import WatermarkDetector

HEX = "0123456789abcdefABCDEF"

def load_key(src: str) -> bytes:
    s = src.strip()
    if len(s) == 64 and all(c in HEX for c in s):
        return bytes.fromhex(s)
    return open(os.path.expanduser(s), "rb").read()

class RxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EchoSeal – Verifier")
        self.resizable(False, False)
        ttk.Style(self).theme_use("clam")

        frm = ttk.Frame(self, padding=16); frm.grid()

        ttk.Label(frm, text="XChaCha20 key (64 hex) or file:").grid(sticky="w")
        self.key_var = tk.StringVar(value="0"*64)
        ttk.Entry(frm, width=60, textvariable=self.key_var).grid(row=0, column=1, pady=4)

        ttk.Label(frm, text="Audio file:").grid(row=1, column=0, sticky="w")
        self.file_var = tk.StringVar()
        ttk.Entry(frm, width=60, textvariable=self.file_var).grid(row=1, column=1, pady=4)
        ttk.Button(frm, text="Browse…", command=self._pick).grid(row=1, column=2, padx=4)

        self.verify_btn = ttk.Button(frm, text="Verify", command=self._verify)
        self.verify_btn.grid(row=2, columnspan=3, pady=10)

        self.status = ttk.Label(frm, text="Awaiting file", font=("Helvetica", 11, "bold"))
        self.status.grid(row=3, columnspan=3, pady=(6,0))

        self._centre()

    # UI helpers
    def _pick(self):
        path = filedialog.askopenfilename(
            title="Open audio", filetypes=[("Audio", "*.wav *.flac *.ogg *.m4a *.mp3"), ("All","*.*")]
        )
        if path: self.file_var.set(path)

    # verification
    def _verify(self):
        path = self.file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("No file", "Select a valid audio file."); return
        try:
            key = load_key(self.key_var.get())
            if len(key) != 32:
                raise ValueError("Need 64-hex chars / 32-byte key")
        except Exception as e:
            messagebox.showerror("Key error", str(e)); return

        try:
            data, fs = sf.read(path, always_2d=False)
        except Exception as e:
            messagebox.showerror("Read error", str(e)); return

        self.verify_btn["state"] = "disabled"; self.status.config(text="Checking…", foreground="black")
        self.update_idletasks()

        ok = WatermarkDetector(key).verify(data, fs)

        self.verify_btn["state"] = "normal"
        if ok:
            self.status.config(text="✅  Authentic", foreground="green")
        else:
            self.status.config(text="⚠️  Tampered / No watermark", foreground="red")

    def _centre(self):
        self.update_idletasks()
        w,h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth()-w)//2
        y = (self.winfo_screenheight()-h)//2
        self.geometry(f"+{x}+{y}")

if __name__ == "__main__":
    RxGUI().mainloop()
