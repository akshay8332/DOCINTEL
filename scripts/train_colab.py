"""Start the full DocILE run. The training loop lives in experiments/multilabel/train.py."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "experiments")]
if len(sys.argv) == 1:
    sys.argv.append(str(ROOT / "configs" / "colab_full_training.yaml"))
runpy.run_path(str(ROOT / "experiments" / "multilabel" / "train.py"), run_name="__main__")
