"""
RT-Watermark-RX – Tkinter GUI
• open audio file
• provide AES key
• Verify → shows Authentic / Tampered
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os, soundfile as sf
from rtwm.detector import WatermarkDetector


def load_key(path_or_hex: str) -> bytes:
    if all(c in "0123456789abcdefABCDEF" for c in path_or_hex.strip()) and len(
        path_or_hex.strip()
    ) in (32, 48, 64):
        return bytes.fromhex(path_or_hex.strip())
    return open(os.path.expanduser(path_or_hex), "rb").read()


class RxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RT-Watermark RX")
        self.resizable(False, False)

        frm = ttk.Frame(self, padding=12)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="AES-GCM Key (hex or file):").grid(sticky="w")
        self.key_var = tk.StringVar(value="00112233445566778899aabbccddeeff")
        ttk.Entry(frm, width=50, textvariable=self.key_var).grid(row=0, column=1, pady=2)

        ttk.Label(frm, text="Audio file:").grid(row=1, column=0, sticky="w")
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(frm, width=50, textvariable=self.file_var)
        file_entry.grid(row=1, column=1, pady=2)
        ttk.Button(frm, text="Browse…", command=self._browse).grid(row=1, column=2, padx=4)

        ttk.Button(frm, text="Verify", command=self._verify).grid(row=2, columnspan=3, pady=6)

        self.result = ttk.Label(frm, text="Awaiting file", font=("Helvetica", 12, "bold"))
        self.result.grid(row=3, columnspan=3, pady=(8, 0))

        self.detector: WatermarkDetector | None = None

    # --------------------------------------------------------------------- ui helpers
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("audio", "*.wav *.flac *.aiff *.ogg *.mp3"), ("all", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def _verify(self):
        path = self.file_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("No file", "Please select a valid audio file")
            return
        try:
            key = load_key(self.key_var.get())
        except Exception as e:
            messagebox.showerror("Key error", f"Failed to load key: {e}")
            return
        if self.detector is None or self.detector._aes is None or key != self.detector._aes._AESCipher__aes._key:  # noqa
            self.detector = WatermarkDetector(key)

        data, fs = sf.read(path, always_2d=False)
        if fs != 48_000:
            messagebox.showinfo(
                "Resample required",
                "For MVP please provide 48 kHz audio (your file is "
                f"{fs/1000:.1f} kHz).",
            )
            return

        ok = self.detector.verify(data)
        self.result["text"] = "✅  Authentic" if ok else "⚠️  Tampered / No Watermark"
        self.result["foreground"] = "green" if ok else "red"


if __name__ == "__main__":
    RxGUI().mainloop()
