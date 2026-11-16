"""Optional text fixes. The DocILE OCR files are left unchanged."""

from __future__ import annotations

import re

_PO_BOX = re.compile(r"\bP\.0\.?(?=\s*BOX\b)", re.IGNORECASE)
_DIGITISH = re.compile(r"[IlO0-9][IlO0-9-]{3,}")


def _fix_token(token: str) -> str:
    if not _DIGITISH.fullmatch(token):
        return token
    return "".join("1" if char in "Il" else "0" if char == "O" else char for char in token)


def correct_common_ocr(text: str) -> str:
    """Fix P.0./I/l/O confusions without adding words the OCR never returned."""
    corrected = _PO_BOX.sub("P.O.", text)
    return "\n".join(
        " ".join(_fix_token(token) for token in line.split(" "))
        for line in corrected.split("\n")
    )
