"""Start the Kaggle party/address fine-tune.

The production trainer is scripts/train_colab.py. This script does not call it.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "packages"), str(ROOT / "experiments")]
runpy.run_path(str(ROOT / "experiments" / "multilabel" / "party_experiment.py"), run_name="__main__")
