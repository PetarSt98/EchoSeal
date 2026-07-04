"""
EchoSeal v0.2 — offline watermark verifier (GUI).
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import soundfile as sf

from gui.common import (
    VERDICT_SUBLINE,
    format_report,
    load_key,
    random_key_hex,
)
from rtwm.detector import Report, WatermarkDetector


class RxGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("EchoSeal v0.2 — Verifier")
        self.minsize(560, 480)
        ttk.Style(self).theme_use("clam")

        self._worker: threading.Thread | None = None

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(6, weight=1)

        ttk.Label(
            root,
            text="Offline verification — authentic / tampered / no watermark",
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        ttk.Label(root, text="Key (64 hex chars or file):").grid(row=1, column=0, sticky="w")
        self.key_var = tk.StringVar()
        key_row = ttk.Frame(root)
        key_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Entry(key_row, textvariable=self.key_var, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(key_row, text="File…", command=self._pick_key, width=6).pack(side="left", padx=(4, 0))
        ttk.Button(key_row, text="Random", command=self._gen_key, width=7).pack(side="left", padx=(4, 0))

        ttk.Label(root, text="Audio file:").grid(row=2, column=0, sticky="w")
        self.file_var = tk.StringVar()
        file_row = ttk.Frame(root)
        file_row.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Entry(file_row, textvariable=self.file_var, width=48).pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Browse…", command=self._pick_audio, width=8).pack(side="left", padx=(4, 0))

        self.verify_btn = ttk.Button(root, text="Verify", command=self._verify)
        self.verify_btn.grid(row=3, column=0, columnspan=3, pady=12)

        verdict_frm = ttk.LabelFrame(root, text="Verdict", padding=10)
        verdict_frm.grid(row=4, column=0, columnspan=3, sticky="ew")
        self.verdict_lbl = ttk.Label(
            verdict_frm,
            text="Awaiting file",
            font=("Segoe UI", 16, "bold"),
        )
        self.verdict_lbl.pack(anchor="w")
        self.sub_lbl = ttk.Label(verdict_frm, text="", wraplength=500)
        self.sub_lbl.pack(anchor="w", pady=(4, 0))

        detail_frm = ttk.LabelFrame(root, text="Details", padding=8)
        detail_frm.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        detail_frm.rowconfigure(0, weight=1)
        detail_frm.columnconfigure(0, weight=1)
        self.detail = tk.Text(detail_frm, height=12, wrap="word", font=("Consolas", 10), state="disabled")
        scroll = ttk.Scrollbar(detail_frm, command=self.detail.yview)
        self.detail.configure(yscrollcommand=scroll.set)
        self.detail.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self._centre()

    def _pick_key(self) -> None:
        path = filedialog.askopenfilename(title="Open key file", filetypes=[("Key", "*.*")])
        if path:
            self.key_var.set(path)

    def _gen_key(self) -> None:
        self.key_var.set(random_key_hex())

    def _pick_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Open audio",
            filetypes=[
                ("Audio", "*.wav *.flac *.ogg *.aiff *.aif"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.file_var.set(path)

    def _set_detail(self, text: str) -> None:
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", text)
        self.detail.configure(state="disabled")

    def _verify(self) -> None:
        if self._worker and self._worker.is_alive():
            return

        path = self.file_var.get().strip()
        if not os.path.isfile(path):
            messagebox.showerror("No file", "Select a valid audio file.")
            return
        try:
            key = load_key(self.key_var.get())
            if len(key) != 32:
                raise ValueError("Key must be 256 bits (64 hex characters or 32 raw bytes).")
        except Exception as exc:
            messagebox.showerror("Key error", str(exc))
            return

        self.verify_btn["state"] = "disabled"
        self.verdict_lbl.config(text="Checking…", foreground="black")
        self.sub_lbl.config(text="")
        self._set_detail(f"Analyzing {os.path.basename(path)} …")

        self._worker = threading.Thread(
            target=self._run_analysis,
            args=(key, path),
            daemon=True,
        )
        self._worker.start()

    def _run_analysis(self, key: bytes, path: str) -> None:
        try:
            data, fs = sf.read(path, always_2d=False)
            report = WatermarkDetector(key).analyze(data, fs)
            self.after(0, lambda: self._show_report(report))
        except Exception as exc:
            self.after(0, lambda: self._show_error(str(exc)))

    def _show_report(self, report: Report) -> None:
        headline, color, lines = format_report(report)
        self.verdict_lbl.config(text=headline, foreground=color)
        self.sub_lbl.config(text=VERDICT_SUBLINE[report.verdict])
        self._set_detail("\n".join(lines))
        self.verify_btn["state"] = "normal"

    def _show_error(self, message: str) -> None:
        self.verdict_lbl.config(text="ERROR", foreground="#cf222e")
        self.sub_lbl.config(text=message)
        self._set_detail("")
        self.verify_btn["state"] = "normal"

    def _centre(self) -> None:
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"+{x}+{y}")


def main() -> None:
    RxGUI().mainloop()


if __name__ == "__main__":
    main()
