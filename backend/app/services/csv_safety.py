"""CSV export helpers that neutralize spreadsheet formulas."""

from __future__ import annotations

import csv
import io
from typing import Iterable


def safe_csv_filename(filename: str | None) -> str:
    """Return a display-only CSV basename. Never use the result as a filesystem path."""
    raw = filename or ""
    name = raw.replace("\\", "/").split("/")[-1]
    name = "".join(character for character in name if character.isprintable() and character not in "\r\n\x00")
    name = name.strip().strip(".")
    if not name.lower().endswith(".csv") or name.lower() == ".csv":
        return ""
    return name[:120]


def safe_download_token(value: str) -> str:
    token = "".join(character for character in value if character.isalnum() or character == "-")
    return token[:80] or "export"


def sanitize_cell(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def rows_to_csv(headers: list[str], rows: Iterable[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: sanitize_cell(row.get(key)) for key in headers})
    return buffer.getvalue()
