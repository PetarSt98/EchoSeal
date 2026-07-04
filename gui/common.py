"""Shared helpers for EchoSeal TX/RX GUIs."""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtwm.detector import Report

HEX_CHARS = "0123456789abcdefABCDEF"

VERDICT_HEADLINE = {
    "authentic": "AUTHENTIC",
    "tampered": "TAMPERED",
    "no-watermark": "NO WATERMARK",
}

VERDICT_SUBLINE = {
    "authentic": "Watermark verified — timeline consistent.",
    "tampered": "Watermark present but inconsistent (possible edit).",
    "no-watermark": "Nothing decodable — wrong key, no mark, or too degraded.",
}

VERDICT_COLOR = {
    "authentic": "#1a7f37",
    "tampered": "#b54708",
    "no-watermark": "#8b949e",
}


def load_key(src: str) -> bytes:
    s = src.strip()
    if len(s) == 64 and all(c in HEX_CHARS for c in s):
        return bytes.fromhex(s)
    path = os.path.expanduser(s)
    with open(path, "rb") as f:
        return f.read()


def random_key_hex() -> str:
    return secrets.token_hex(32)


def format_report(report: Report) -> tuple[str, str, list[str]]:
    """Return headline, hex colour, and detail lines for the RX panel."""
    lines: list[str] = []
    if report.hits:
        first, last = report.hits[0], report.hits[-1]
        lines.append(
            f"Frames decoded: {len(report.hits)} "
            f"(counters {first.ctr}..{last.ctr}, coverage {report.coverage:.0%})"
        )
        lines.append(
            f"Time span: {first.start / 48_000:.2f}s .. {last.start / 48_000:.2f}s"
        )
    else:
        lines.append("Frames decoded: 0")

    if report.issues:
        lines.append("")
        lines.append("Issues:")
        lines.extend(f"  • {issue}" for issue in report.issues)

    if report.unverified_spans:
        lines.append("")
        label = "Unverified spans" if report.verdict == "tampered" else "Degraded spans (not tampering)"
        lines.append(f"{label}:")
        for t0, t1 in report.unverified_spans:
            lines.append(f"  • {t0:.2f}s .. {t1:.2f}s ({t1 - t0:.2f}s)")

    if report.verdict == "authentic" and not report.issues and not report.unverified_spans:
        lines.append("")
        lines.append("No inconsistencies detected.")

    headline = VERDICT_HEADLINE[report.verdict]
    return headline, VERDICT_COLOR[report.verdict], lines
