#!/usr/bin/env python
"""Download and extract the FUNSD dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without install: add packages/ to path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages"))

from docintel.data.funsd import download_funsd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FUNSD")
    parser.add_argument(
        "--dest",
        type=Path,
        default=_ROOT / "data" / "funsd",
        help="Destination directory (default: ./data/funsd)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    path = download_funsd(args.dest, force=args.force)
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
