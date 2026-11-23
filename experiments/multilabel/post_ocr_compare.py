"""Compare frozen address predictions with selective text corrections.

The epoch-3 checkpoint is not loaded. DocILE OCR files, the threshold, and the
grouping are not changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multilabel.address_reader_compare import text_inside
from multilabel.checkpoint import assert_diagnostic_destination
from multilabel.post_ocr import (
    correct_prediction,
    normalize_prediction,
    refine_boundary,
    same_wording,
    selective_fallback,
    sequence_distance,
    stack_correction,
    wording_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("vendor_address", "customer_billing_address")
ARMS = ("baseline", "normalize", "correct", "boundary", "fallback", "stacked")


def _levenshtein(left: str, right: str) -> int:
    from multilabel.post_ocr import _levenshtein as distance

    return distance(left, right)


def apply_arm(arm: str, prediction: str | None, readers: list[str]) -> str | None:
    if prediction is None:
        return None
    if arm == "baseline":
        return prediction
    if arm == "normalize":
        return normalize_prediction(prediction)
    if arm == "correct":
        return correct_prediction(prediction)
    if arm == "boundary":
        return refine_boundary(prediction)
    if arm == "fallback":
        return selective_fallback(prediction, readers)
    if arm == "stacked":
        return stack_correction(prediction, readers)
    raise KeyError(arm)


def _rates(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def score_arm(rows: list[dict], arm: str) -> dict:
    per_field = {}
    for name in FIELDS:
        subset = [row for row in rows if row["field"] == name]
        golds = [row for row in subset if not row["unmatched"]]
        extras = [row for row in subset if row["unmatched"]]
        exact_tp = wording_tp = 0
        exact_fp = wording_fp = 0
        char_errors = char_total = 0
        word_errors = word_total = 0
        located_char_errors = located_char_total = 0
        located_word_errors = located_word_total = 0
        wording_errors = wording_total = 0
        for row in golds:
            text = row["texts"][arm]
            gold = row["gold"] or ""
            located = row["overlap"] >= 0.5 and text is not None
            exact = bool(located and text == gold)
            wording = bool(located and same_wording(text or "", gold))
            exact_tp += int(exact)
            wording_tp += int(wording)
            if text is not None and not exact:
                exact_fp += 1
            if text is not None and not wording:
                wording_fp += 1
            compared = text or ""
            char_errors += _levenshtein(compared, gold)
            char_total += len(gold)
            gold_words = (gold or "").split()
            pred_words = compared.split()
            word_errors += sequence_distance(pred_words, gold_words)
            word_total += len(gold_words)
            gold_wording = wording_tokens(gold)
            wording_errors += sequence_distance(wording_tokens(compared), gold_wording)
            wording_total += len(gold_wording)
            if located:
                located_char_errors += _levenshtein(compared, gold)
                located_char_total += len(gold)
                located_word_errors += sequence_distance(pred_words, gold_words)
                located_word_total += len(gold_words)
        for row in extras:
            if row["texts"][arm] is not None:
                exact_fp += 1
                wording_fp += 1
        exact = _rates(exact_tp, exact_fp, len(golds) - exact_tp)
        wording = _rates(wording_tp, wording_fp, len(golds) - wording_tp)
        per_field[name] = {
            "gold": len(golds),
            "exact_matches": exact_tp,
            "wording_matches": wording_tp,
            "exact": exact,
            "wording": wording,
            "cer": char_errors / char_total if char_total else None,
            "wer": word_errors / word_total if word_total else None,
            "located_cer": located_char_errors / located_char_total if located_char_total else None,
            "located_wer": located_word_errors / located_word_total if located_word_total else None,
            "wording_wer": wording_errors / wording_total if wording_total else None,
        }
    return per_field


def _examples(rows: list[dict], arm: str) -> dict:
    gained = []
    wording_lost = []
    for row in rows:
        if row["unmatched"] or row["overlap"] < 0.5:
            continue
        baseline = row["texts"]["baseline"]
        edited = row["texts"][arm]
        gold = row["gold"]
        if edited == gold and baseline != gold and len(gained) < 3:
            gained.append(
                {
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "field": row["field"],
                    "gold": gold,
                    "baseline": baseline,
                    "corrected": edited,
                }
            )
        if same_wording(baseline or "", gold or "") and not same_wording(edited or "", gold or "") and len(wording_lost) < 3:
            wording_lost.append(
                {
                    "document_id": row["document_id"],
                    "page": row["page"],
                    "field": row["field"],
                    "gold": gold,
                    "baseline": baseline,
                    "corrected": edited,
                }
            )
    return {"exact_gained": gained, "wording_lost": wording_lost}


def build_rows(audit: dict, jobs: dict, readers: dict[str, dict[str, list[dict]]]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for case in audit["cases"]:
        if case["unmatched_prediction"]:
            continue
        grouped.setdefault((case["document_id"], case["page"]), []).append(case)
    extras = [case for case in audit["cases"] if case["unmatched_prediction"]]
    rows = []
    for page in jobs["pages"]:
        page_readers = {
            name: pages.get(page["document_id"], []) for name, pages in readers.items()
        }
        pending = list(grouped.get((page["doc_id"], page["page"]), []))
        for field in page["fields"]:
            case = next(
                (
                    item
                    for item in pending
                    if item["field"] == field["field"] and item["gold"] == field["text"]
                ),
                None,
            )
            if case is None:
                raise RuntimeError(f"No audit prediction for {page['doc_id']} page {page['page']} {field['field']}")
            pending.remove(case)
            prediction = case.get("prediction")
            overlap = float(case.get("overlap") or 0.0)
            reader_texts = [
                text_inside(page_readers[name], field["bbox"]) for name in ("doctr", "ppocrv5") if name in page_readers
            ]
            texts = {arm: apply_arm(arm, prediction, reader_texts) for arm in ARMS}
            rows.append(
                {
                    "document_id": page["doc_id"],
                    "page": page["page"],
                    "field": field["field"],
                    "gold": field["text"],
                    "overlap": overlap,
                    "score": case.get("score"),
                    "readers": reader_texts,
                    "unmatched": False,
                    "texts": texts,
                }
            )
    for case in extras:
        prediction = case.get("prediction")
        texts = {arm: apply_arm(arm, prediction, []) for arm in ARMS}
        rows.append(
            {
                "document_id": case["document_id"],
                "page": case["page"],
                "field": case["field"],
                "gold": None,
                "overlap": 0.0,
                "score": case.get("score"),
                "readers": [],
                "unmatched": True,
                "texts": texts,
            }
        )
    return rows


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_report(report: dict) -> str:
    lines = [
        "# Post-OCR correction",
        "",
        "Frozen baseline: epoch-3 predictions, DocILE OCR, threshold 0.5, line-block-v2.",
        "The checkpoint was not loaded. DocILE files, the threshold, and the grouping were not changed.",
        "",
        "Exact match is the official address string. Wording match ignores punctuation and line breaks.",
        "A correction is useful only when exact match rises and wording match does not fall.",
        "",
        "Normalize tightens spaces. Correct applies P.0.→P.O., digit I/l/O, and a clo→c/o token.",
        "Boundary drops a repeated line or a lone leading digit. Fallback replaces a token only when docTR and PP-OCRv5 agree and the edit is one or two characters.",
        "Stacked runs correct, then boundary, then fallback.",
        "",
        "| Arm | Field | Exact | Wording | Exact precision | Exact recall | Wording precision | Wording recall | CER | WER | Wording WER |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for name in FIELDS:
            row = report["arms"][arm][name]
            lines.append(
                "| {arm} | {name} | {exact} | {wording} | {ep} | {er} | {wp} | {wr} | {cer} | {wer} | {wwer} |".format(
                    arm=arm,
                    name=name,
                    exact=row["exact_matches"],
                    wording=row["wording_matches"],
                    ep=_fmt(row["exact"]["precision"]),
                    er=_fmt(row["exact"]["recall"]),
                    wp=_fmt(row["wording"]["precision"]),
                    wr=_fmt(row["wording"]["recall"]),
                    cer=_fmt(row["cer"]),
                    wer=_fmt(row["wer"]),
                    wwer=_fmt(row["wording_wer"]),
                )
            )
    lines.extend(["", "## Changes from the frozen predictions", ""])
    for arm in ARMS:
        if arm == "baseline":
            continue
        lines.append(f"### {arm}")
        lines.append("")
        samples = report["examples"][arm]
        if samples["exact_gained"]:
            lines.append("Exact matches gained:")
            lines.append("")
            for sample in samples["exact_gained"]:
                lines.append(f"- `{sample['document_id']}` page {sample['page']} {sample['field']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - baseline: {sample['baseline']!r}")
                lines.append(f"  - corrected: {sample['corrected']!r}")
            lines.append("")
        if samples["wording_lost"]:
            lines.append("Wording matches lost:")
            lines.append("")
            for sample in samples["wording_lost"]:
                lines.append(f"- `{sample['document_id']}` page {sample['page']} {sample['field']}")
                lines.append(f"  - gold: {sample['gold']!r}")
                lines.append(f"  - baseline: {sample['baseline']!r}")
                lines.append(f"  - corrected: {sample['corrected']!r}")
            lines.append("")
        if not samples["exact_gained"] and not samples["wording_lost"]:
            lines.append("No exact-match gains and no wording losses on the located addresses.")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json",
    )
    parser.add_argument(
        "--jobs",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-reader-jobs-100.json",
    )
    parser.add_argument(
        "--doctr",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-reader-doctr-crops.jsonl",
    )
    parser.add_argument(
        "--ppocr",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-reader-ppocrv5-crops.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "diagnostics" / "address-post-ocr-100.json",
    )
    args = parser.parse_args()
    assert_diagnostic_destination(args.output)
    from multilabel.address_reader_compare import _load_reader

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    readers = {"doctr": _load_reader(args.doctr)[1], "ppocrv5": _load_reader(args.ppocr)[1]}
    rows = build_rows(audit, jobs, readers)
    report = {
        "baseline": "epoch-3 + DocILE OCR + threshold 0.5 + line-block-v2",
        "checkpoint_updated": False,
        "ocr_replaced": False,
        "threshold": 0.5,
        "grouping": "line-block-v2",
        "validation_pages": audit["validation_pages"],
        "documents": audit["documents"],
        "arms": {arm: score_arm(rows, arm) for arm in ARMS},
        "examples": {arm: _examples(rows, arm) for arm in ARMS if arm != "baseline"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = args.output.with_suffix(".md")
    summary.write_text(render_report(report), encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
