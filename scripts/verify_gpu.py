"""Print the PyTorch, CUDA, and GPU details before a Colab training run."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel.device import describe_device


def main() -> int:
    require_gpu = "--require-gpu" in sys.argv
    try:
        device = describe_device(require_gpu=require_gpu)
    except RuntimeError as exc:
        print(exc, flush=True)
        return 1
    print(f"Selected device: {device}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
