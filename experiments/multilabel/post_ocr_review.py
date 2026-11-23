"""Ablation, word-change precision, and a development confidence check.

No new correction rules are added. The official test set is not scored: its public
annotations do not contain field text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from multilabel.checkpoint import assert_diagnostic_destination
from multilabel.post_ocr import apply_layers, same_wording, word_change_outcomes, wording_edit_count, wording_tokens
from multilabel.post_ocr_compare import build_rows

ROOT = Path(__file__).resolve().parents[2]
COMBOS = (
    (),
    ("correct",),
    ("boundary",),
    ("fallback",),
    ("correct", "boundary"),
    ("correct", "fallback"),
    ("boundary", "fallback"),
    ("correct", "boundary", "fallback"),
)
STACK = ("correct", "boundary", "fallback")


def _name(layers: tuple[str, ...]) -> str:
    return "+".join(layers) if layers else "baseline"


def _located(row: dict) -> bool:
    return not row["unmatched"] and row["overlap"] >= 0.5 and row["texts"]["baseline"] is not None


def _combo_text(row: dict, layers: tuple[str, ...]) -> str | None:
    baseline = row["texts"]["baseline"]
    if baseline is None:
        return None
    return apply_layers(baseline, row.get("readers") or [], layers)


def ablation(rows: list[dict]) -> dict:
    report = {}
    for layers in COMBOS:
        name = _name(layers)
        exact = wording = exact_gained = exact_lost = wording_gained = wording_lost = 0
        golds = 0
        changes = {"improved": 0, "harmed": 0, "neutral": 0}
        for row in rows:
            if row["unmatched"]:
                continue
            golds += 1
            baseline = row["texts"]["baseline"]
            edited = _combo_text(row, layers)
            gold = row["gold"] or ""
            located = _located(row)
            if located and edited == gold:
                exact += 1
            if located and same_wording(edited or "", gold):
                wording += 1
            if not located:
                continue
            base_exact = baseline == gold
            new_exact = edited == gold
            exact_gained += int(new_exact and not base_exact)
            exact_lost += int(base_exact and not new_exact)
            base_wording = same_wording(baseline or "", gold)
            new_wording = same_wording(edited or "", gold)
            wording_gained += int(new_wording and not base_wording)
            wording_lost += int(base_wording and not new_wording)
            for outcome in word_change_outcomes(baseline or "", edited or "", gold):
                changes[outcome] += 1
        decisive = changes["improved"] + changes["harmed"]
        report[name] = {
            "layers": list(layers),
            "gold": golds,
            "exact_matches": exact,
            "wording_matches": wording,
            "exact_gained": exact_gained,
            "exact_lost": exact_lost,
            "wording_gained": wording_gained,
            "wording_lost": wording_lost,
            "word_changes": changes,
            "change_precision": changes["improved"] / decisive if decisive else None,
            "changes_that_improve": changes["improved"],
            "changes_that_harm": changes["harmed"],
        }
    return report


def _features(row: dict, corrected: str) -> list[float]:
    baseline = row["texts"]["baseline"] or ""
    readers = row.get("readers") or ["", ""]
    while len(readers) < 2:
        readers.append("")
    baseline_tokens = wording_tokens(baseline)
    corrected_tokens = wording_tokens(corrected)
    return [
        float(row.get("score") or 0.0),
        float(corrected != baseline),
        float(wording_edit_count(baseline, corrected)),
        float(same_wording(readers[0], readers[1])),
        float(same_wording(readers[0], corrected) and same_wording(readers[1], corrected)),
        float(len(corrected_tokens)),
        float(same_wording(baseline, corrected)),
        float(len(baseline_tokens)),
    ]


def _curve(labels: list[int], scores: list[float]) -> list[dict]:
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    rows = []
    correct = 0
    total = len(labels)
    for count, index in enumerate(order, start=1):
        correct += labels[index]
        if count == total or count % max(1, total // 10) == 0:
            rows.append(
                {
                    "approved": count,
                    "coverage": count / total if total else None,
                    "precision": correct / count,
                    "errors_released": count - correct,
                    "left_for_a_person": total - count,
                }
            )
    return rows


def _coverage_at(labels: list[int], scores: list[float], target: float) -> dict | None:
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    correct = 0
    best = None
    for count, index in enumerate(order, start=1):
        correct += labels[index]
        precision = correct / count
        if precision + 1e-12 >= target:
            best = {
                "approved": count,
                "coverage": count / len(labels) if labels else None,
                "precision": precision,
                "errors_released": count - correct,
                "left_for_a_person": len(labels) - count,
            }
    return best


def calibrate(rows: list[dict]) -> dict:
    located = [row for row in rows if _located(row)]
    corrected = [_combo_text(row, STACK) or "" for row in located]
    labels = [int(text == (row["gold"] or "")) for row, text in zip(located, corrected)]
    matrix = np.array([_features(row, text) for row, text in zip(located, corrected)], dtype=float)
    groups = np.array([row["document_id"] for row in located])
    probabilities = np.zeros(len(located), dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(matrix, labels, groups):
        if len(set(labels[index] for index in train)) < 2:
            probabilities[test] = float(np.mean([labels[index] for index in train]))
            continue
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=500)),
            ]
        )
        pipeline.fit(matrix[train], [labels[index] for index in train])
        probabilities[test] = pipeline.predict_proba(matrix[test])[:, 1]
    model_scores = [float(row.get("score") or 0.0) for row in located]
    missed = sum(1 for row in rows if not row["unmatched"] and not _located(row))
    targets = {}
    for target in (0.99, 0.97, 0.95):
        found = _coverage_at(labels, probabilities.tolist(), target)
        targets[f"{target:.2f}"] = found
    return {
        "evaluated_on": "5-fold document holdout inside the 100 validation invoices",
        "official_test_scored": False,
        "official_test_reason": "Public test annotations contain page metadata only, so field text cannot be scored.",
        "located_addresses": len(located),
        "missed_addresses_left_for_a_person": missed,
        "exact_matches": int(sum(labels)),
        "feature_names": [
            "model_score",
            "text_changed",
            "wording_edits",
            "readers_agree",
            "readers_match_corrected",
            "corrected_token_count",
            "wording_unchanged",
            "baseline_token_count",
        ],
        "label": "corrected text equals the official address",
        "cross_validated_curve": _curve(labels, probabilities.tolist()),
        "model_score_curve": _curve(labels, model_scores),
        "highest_coverage_at_precision": targets,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def render_ablation(report: dict) -> str:
    lines = [
        "# Correction ablation",
        "",
        "Development set: the same 100 validation invoices. No new correction rule was added.",
        "Exact match is the official string. Wording match ignores punctuation and line breaks.",
        "Change precision counts wording-token edits that move closer to the official text, among edits that move either closer or farther. Neutral edits stay the same distance away.",
        "",
        "| Layers | Exact | Wording | Exact gained | Exact lost | Wording gained | Wording lost | Edits closer | Edits farther | Change precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["arms"].items():
        lines.append(
            "| {name} | {exact} | {wording} | {gained} | {lost} | {wgained} | {wlost} | {closer} | {farther} | {precision} |".format(
                name=name,
                exact=row["exact_matches"],
                wording=row["wording_matches"],
                gained=row["exact_gained"],
                lost=row["exact_lost"],
                wgained=row["wording_gained"],
                wlost=row["wording_lost"],
                closer=row["changes_that_improve"],
                farther=row["changes_that_harm"],
                precision=_fmt(row["change_precision"]),
            )
        )
    lines.extend(
        [
            "",
            f"Gold addresses: {next(iter(report['arms'].values()))['gold']}.",
            "Layers run in this order when combined: character correction, boundary cleanup, reader agreement.",
            "",
        ]
    )
    return "\n".join(lines)


def render_confidence(report: dict) -> str:
    lines = [
        "# Address confidence",
        "",
        "Development check only. Each address is scored by a model that did not train on its invoice.",
        "The question is the probability that the full corrected address matches the official string.",
        "Public test annotations have no field text, so this is not an official test score and not a measured review-time saving.",
        "",
        f"Located addresses: {report['located_addresses']}. Exact after correction: {report['exact_matches']}.",
        f"Addresses with no located prediction, left for a person: {report['missed_addresses_left_for_a_person']}.",
        "",
        "## Cross-validated probability",
        "",
        "| Approved | Coverage | Precision | Errors released | Left for a person |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["cross_validated_curve"]:
        lines.append(
            f"| {row['approved']} | {_pct(row['coverage'])} | {_fmt(row['precision'])} | {row['errors_released']} | {row['left_for_a_person']} |"
        )
    lines.extend(
        [
            "",
            "## Model score alone, ranking the same corrected addresses",
            "",
            "| Approved | Coverage | Precision | Errors released | Left for a person |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["model_score_curve"]:
        lines.append(
            f"| {row['approved']} | {_pct(row['coverage'])} | {_fmt(row['precision'])} | {row['errors_released']} | {row['left_for_a_person']} |"
        )
    lines.extend(["", "## Highest coverage that holds a precision target", ""])
    for target, found in report["highest_coverage_at_precision"].items():
        if found is None:
            lines.append(f"- Precision {target}: no automatic set holds this precision.")
        else:
            lines.append(
                f"- Precision {target}: approve {found['approved']} of {report['located_addresses']} "
                f"({_pct(found['coverage'])}), errors released {found['errors_released']}, "
                f"located addresses left for a person {found['left_for_a_person']}."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json")
    parser.add_argument("--jobs", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-reader-jobs-100.json")
    parser.add_argument("--doctr", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-reader-doctr-crops.jsonl")
    parser.add_argument("--ppocr", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-reader-ppocrv5-crops.jsonl")
    parser.add_argument("--ablation", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-correction-ablation-100.json")
    parser.add_argument("--confidence", type=Path, default=ROOT / "artifacts" / "diagnostics" / "address-confidence-100.json")
    args = parser.parse_args()
    assert_diagnostic_destination(args.ablation)
    assert_diagnostic_destination(args.confidence)
    from multilabel.address_reader_compare import _load_reader

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    readers = {"doctr": _load_reader(args.doctr)[1], "ppocrv5": _load_reader(args.ppocr)[1]}
    rows = build_rows(audit, jobs, readers)
    ablation_report = {
        "development_pages": 100,
        "official_test_scored": False,
        "new_rules_added": False,
        "checkpoint_updated": False,
        "arms": ablation(rows),
    }
    confidence_report = calibrate(rows)
    args.ablation.parent.mkdir(parents=True, exist_ok=True)
    args.ablation.write_text(json.dumps(ablation_report, indent=2), encoding="utf-8")
    args.ablation.with_suffix(".md").write_text(render_ablation(ablation_report), encoding="utf-8")
    args.confidence.write_text(json.dumps(confidence_report, indent=2), encoding="utf-8")
    args.confidence.with_suffix(".md").write_text(render_confidence(confidence_report), encoding="utf-8")
    print(args.ablation.with_suffix(".md").read_text(encoding="utf-8"))
    print(args.confidence.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
