"""Run a saved multi-label checkpoint and score it with the DocILE evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel.data import encode_document, load_settings, prepare_documents, write_json
from multilabel.evaluate import load_multilabel, official_kile, predict_pages


def main() -> int:
    settings = load_settings(ROOT / "configs" / "multilabel-200.yaml")
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(settings["output_dir"]) / "best"
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model, processor, _payload = load_multilabel(checkpoint, device)
    docs = prepare_documents(Path(settings["data_dir"]), "val", int(settings["eval_limit"]))
    pages = [encode_document(doc, processor, max_length=int(settings["max_seq_length"])) for doc in docs]
    predicted = predict_pages(model, pages, settings, device)
    predicted["official"] = official_kile(settings, pages, predicted["predictions"])
    destination = ROOT / "artifacts" / "diagnostics" / "multilabel-eval.json"
    write_json(destination, {"predictions": predicted["predicted_fields"], "official": predicted["official"]})
    write_json(ROOT / "artifacts" / "diagnostics" / "multilabel-val-predictions.json", predicted["predictions"])
    print(f"Wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
