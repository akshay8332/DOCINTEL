"""Offline diagnosis of the remaining address errors.

Reads the epoch-3 probability cache and the stored audits. It does not load a
checkpoint, does not edit a prediction, and does not change production.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

from multilabel import FIELD_NAMES  # noqa: E402
from multilabel.address_text_audit import center_inside, norm_token, tokens  # noqa: E402
from multilabel.post_ocr import (  # noqa: E402
    correct_prediction,
    refine_boundary,
    selective_fallback,
    stack_correction,
    word_change_outcomes,
)

CACHE = ROOT / "artifacts" / "diagnostics" / "address-threshold-word-probs.jsonl"
LABELS = ROOT / "artifacts" / "diagnostics" / "address-error-epoch3-labels.jsonl"
AUDIT = ROOT / "artifacts" / "diagnostics" / "address-text-audit-epoch3-local.json"
PUBLISHED_AUDIT = ROOT / "artifacts" / "diagnostics" / "address-text-audit-100.json"
JOBS = ROOT / "artifacts" / "diagnostics" / "address-reader-jobs-100.json"
OUTPUT = ROOT / "artifacts" / "diagnostics" / "address-remaining-errors.json"
VENDOR = FIELD_NAMES.index("vendor_address")
CUSTOMER = FIELD_NAMES.index("customer_billing_address")
ADDRESS = {"vendor_address": "vendor", "customer_billing_address": "customer"}

STREET_CUES = {
    "st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive",
    "ln", "lane", "way", "suite", "ste", "floor", "fl", "box", "po", "pobox", "hwy",
    "highway", "pkwy", "parkway", "ct", "court", "pl", "place", "cir", "circle",
    "trl", "trail", "ter", "terrace", "plaza", "pike", "route", "rte",
}
STATE_CODES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "dc", "de", "fl", "ga", "hi", "ia", "id",
    "il", "in", "ks", "ky", "la", "ma", "md", "me", "mi", "mn", "mo", "ms", "mt", "nc",
    "nd", "ne", "nh", "nj", "nm", "nv", "ny", "oh", "ok", "or", "pa", "ri", "sc", "sd",
    "tn", "tx", "ut", "va", "vt", "wa", "wi", "wv", "wy",
}
STATE_NAMES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "newhampshire",
    "newjersey", "newmexico", "newyork", "northcarolina", "northdakota", "ohio", "oklahoma",
    "oregon", "pennsylvania", "rhodeisland", "southcarolina", "southdakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "westvirginia", "wisconsin",
    "wyoming", "karnataka",
}
COUNTRIES = {
    "usa", "us", "unitedstates", "unitedstatesofamerica", "america", "india", "kenya",
}
ATTENTION = {"attn", "attention", "att", "co", "clo"}
HEADING = re.compile(
    r"\b(bill to|ship to|sold to|remit to|advertiser|agency|station|invoice|page)\b",
    re.IGNORECASE,
)
PIN = re.compile(r"^\d{5}(?:\d{4})?$|^\d{6}$")


def load_pages() -> list[dict]:
    return [json.loads(line) for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()]


def line_roles(gold_text: str) -> list[list[str]]:
    """Label each official token from its line, without changing any prediction."""
    roles: list[list[str]] = []
    seen_street = False
    for raw_line in (gold_text or "").split("\n"):
        parts = [token for token in tokens(raw_line) if norm_token(token)]
        norms = [norm_token(token) for token in parts]
        if not norms:
            continue
        row = []
        streetish = any(norm in STREET_CUES for norm in norms) or bool(re.match(r"^\d", parts[0]))
        attention = norms[0] in ATTENTION or any(norm in {"ms", "mr", "mrs"} for norm in norms)
        country_line = "america" in norms or "usa" in norms or "india" in norms
        for norm in norms:
            if PIN.fullmatch(norm):
                role = "pin"
            elif norm in {"ms", "mr", "mrs"}:
                role = "attention"
            elif norm in STATE_CODES or norm in STATE_NAMES:
                role = "state"
            elif norm in COUNTRIES or (country_line and norm in {"united", "states", "of"}):
                role = "country"
            elif attention or norm in ATTENTION:
                role = "attention"
            elif streetish:
                role = "street"
            elif not seen_street:
                role = "company"
            else:
                role = "locality"
            row.append(role)
        if streetish:
            seen_street = True
        roles.append(row)
    return roles


def role_for(gold_text: str, ocr_norm: str) -> str:
    """First official token with the same letters. Line rules are descriptive only."""
    for roles, raw_line in zip(
        line_roles(gold_text),
        [line for line in (gold_text or "").split("\n") if tokens(line)],
    ):
        official = [norm_token(token) for token in tokens(raw_line) if norm_token(token)]
        for norm, role in zip(official, roles):
            if norm == ocr_norm:
                return role
    return "unclassified"


def same_line(left: list[int], right: list[int]) -> bool:
    overlap = min(left[3], right[3]) - max(left[1], right[1])
    smaller = max(1, min(left[3] - left[1], right[3] - right[1]))
    return overlap >= 0.4 * smaller


def low_confidence_tokens(pages: list[dict]) -> list[dict]:
    rows = []
    for page in pages:
        words = page["words"]
        for gold in page["golds"]:
            field = gold["fieldtype"]
            key = ADDRESS[field]
            official = {norm_token(token) for token in tokens(gold["text"]) if norm_token(token)}
            in_box = [index for index, word in enumerate(words) if center_inside(gold["bbox"], word["box"])]
            for index in in_box:
                word = words[index]
                norm = norm_token(word["text"])
                if not norm or norm not in official or word[key] >= 0.35:
                    continue
                neighbors = []
                for other in (index - 1, index + 1):
                    if 0 <= other < len(words):
                        neighbors.append(words[other]["text"])
                line = [
                    words[other]["text"]
                    for other in in_box
                    if other != index and same_line(word["box"], words[other]["box"])
                ]
                above = []
                for other, item in enumerate(words):
                    box = item["box"]
                    if box[3] > word["box"][1]:
                        continue
                    if word["box"][1] - box[3] > 40:
                        continue
                    horizontal = min(word["box"][2], box[2]) - max(word["box"][0], box[0])
                    if horizontal <= 0:
                        continue
                    above.append(item["text"])
                height = word["box"][3] - word["box"][1]
                center_x = (word["box"][0] + word["box"][2]) / 2
                rows.append(
                    {
                        "document_id": page["document_id"],
                        "page": page["page"],
                        "field": field,
                        "official_text": gold["text"],
                        "ocr_text": word["text"],
                        "role": role_for(gold["text"], norm),
                        "address_probability": round(word[key], 4),
                        "box": word["box"],
                        "height": height,
                        "center_x": round(center_x, 1),
                        "right_column": center_x >= 500,
                        "neighbors": neighbors,
                        "same_line": line[:8],
                        "above": above[:6],
                        "near_heading": any(HEADING.search(text) for text in above),
                        "first_line": norm in {
                            norm_token(token)
                            for token in tokens((gold["text"] or "").split("\n")[0])
                            if norm_token(token)
                        },
                    }
                )
    return rows


def height_baseline(pages: list[dict]) -> dict:
    heights = {name: [] for name in ADDRESS}
    columns = {name: {"right": 0, "all": 0} for name in ADDRESS}
    roles = {name: Counter() for name in ADDRESS}
    for page in pages:
        for gold in page["golds"]:
            field = gold["fieldtype"]
            key = ADDRESS[field]
            official = {norm_token(token) for token in tokens(gold["text"]) if norm_token(token)}
            for word in page["words"]:
                norm = norm_token(word["text"])
                if not center_inside(gold["bbox"], word["box"]) or norm not in official:
                    continue
                heights[field].append(word["box"][3] - word["box"][1])
                columns[field]["all"] += 1
                if (word["box"][0] + word["box"][2]) / 2 >= 500:
                    columns[field]["right"] += 1
                roles[field][role_for(gold["text"], norm)] += 1
    summary = {}
    for field, values in heights.items():
        ordered = sorted(values)
        summary[field] = {
            "tokens": len(ordered),
            "median_height": ordered[len(ordered) // 2] if ordered else None,
            "p10_height": ordered[max(0, len(ordered) // 10 - 1)] if ordered else None,
            "right_column_share": columns[field]["right"] / columns[field]["all"] if columns[field]["all"] else None,
            "roles": dict(roles[field]),
        }
    return summary


def attach_labels(rows: list[dict]) -> None:
    if not LABELS.is_file():
        return
    lookup = {}
    for line in LABELS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        page = json.loads(line)
        for word in page["words"]:
            lookup[(page["document_id"], page["page"], tuple(word["box"]), word["text"])] = word
    for row in rows:
        found = lookup.get((row["document_id"], row["page"], tuple(row["box"]), row["ocr_text"]))
        if not found:
            continue
        scores = found["scores"]
        best = max(range(len(scores)), key=lambda index: scores[index])
        row["predicted_label"] = FIELD_NAMES[best]
        row["predicted_probability"] = round(scores[best], 4)
        ranked = sorted(
            ((FIELD_NAMES[index], round(score, 4)) for index, score in enumerate(scores)),
            key=lambda item: item[1],
            reverse=True,
        )
        row["top_labels"] = ranked[:3]


def pattern_summary(rows: list[dict], baseline: dict) -> dict:
    summary = {}
    for field in ADDRESS:
        subset = [row for row in rows if row["field"] == field]
        roles = Counter(row["role"] for row in subset)
        labels = Counter(row.get("predicted_label", "unavailable") for row in subset)
        base_roles = baseline[field]["roles"]
        role_rates = {}
        for role, count in roles.items():
            base = base_roles.get(role, 0)
            role_rates[role] = {
                "low_confidence": count,
                "all_official_in_box": base,
                "low_share": count / len(subset) if subset else None,
                "base_share": base / baseline[field]["tokens"] if baseline[field]["tokens"] else None,
            }
        heights = sorted(row["height"] for row in subset)
        summary[field] = {
            "count": len(subset),
            "roles": role_rates,
            "predicted_labels": dict(labels),
            "right_column": sum(row["right_column"] for row in subset),
            "right_column_share": sum(row["right_column"] for row in subset) / len(subset) if subset else None,
            "base_right_column_share": baseline[field]["right_column_share"],
            "near_heading": sum(row["near_heading"] for row in subset),
            "first_line": sum(row["first_line"] for row in subset),
            "median_height": heights[len(heights) // 2] if heights else None,
            "base_median_height": baseline[field]["median_height"],
            "base_p10_height": baseline[field]["p10_height"],
            "at_or_below_base_p10": sum(
                row["height"] <= (baseline[field]["p10_height"] or 0) for row in subset
            ),
        }
    return summary


def audit_causes() -> dict:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    primary = {name: Counter() for name in ADDRESS}
    issues = {name: Counter() for name in ADDRESS}
    exact = Counter()
    gold = Counter()
    for case in audit["cases"]:
        if case.get("unmatched_prediction"):
            continue
        field = case["field"]
        gold[field] += 1
        if case.get("exact"):
            exact[field] += 1
            continue
        primary[field][case.get("primary") or "other"] += 1
        for issue in case.get("issues") or []:
            issues[field][issue] += 1
    return {
        "gold": dict(gold),
        "exact": dict(exact),
        "primary": {field: dict(counter) for field, counter in primary.items()},
        "issues_present": {field: dict(counter) for field, counter in issues.items()},
    }


def named_ocr_cases(pages: list[dict]) -> list[dict]:
    wanted = {
        ("0c0b7d2c4ce74a74ac4ca19c", 0, "vendor_address"),
        ("1141b304c74042f1970279ae", 0, "customer_billing_address"),
        ("1ac88b2f43e5490584a3b864", 0, "vendor_address"),
    }
    found = []
    for page in pages:
        for gold in page["golds"]:
            key = (page["document_id"], page["page"], gold["fieldtype"])
            if key not in wanted:
                continue
            inside = []
            for word in page["words"]:
                if center_inside(gold["bbox"], word["box"]):
                    inside.append(
                        {
                            "text": word["text"],
                            "box": word["box"],
                            "vendor": round(word["vendor"], 4),
                            "customer": round(word["customer"], 4),
                        }
                    )
            found.append(
                {
                    "document_id": page["document_id"],
                    "page": page["page"],
                    "field": gold["fieldtype"],
                    "official_text": gold["text"],
                    "ocr_words_inside_box": inside,
                }
            )
    return found


def stack_layers() -> dict:
    """Replay the published comparison. This is the path that scores 72."""
    from multilabel.address_reader_compare import _load_reader, text_inside
    from multilabel.post_ocr_compare import build_rows

    audit = json.loads(PUBLISHED_AUDIT.read_text(encoding="utf-8"))
    jobs = json.loads(JOBS.read_text(encoding="utf-8"))
    readers = {
        "doctr": _load_reader(ROOT / "artifacts" / "diagnostics" / "address-reader-doctr-crops.jsonl")[1],
        "ppocrv5": _load_reader(ROOT / "artifacts" / "diagnostics" / "address-reader-ppocrv5-crops.jsonl")[1],
    }
    rows = build_rows(audit, jobs, readers)
    combos = {
        "baseline": (),
        "correct": ("correct",),
        "boundary": ("boundary",),
        "fallback": ("fallback",),
        "correct+boundary": ("correct", "boundary"),
        "correct+fallback": ("correct", "fallback"),
        "full": ("correct", "boundary", "fallback"),
    }
    report = {}
    located_rows = [row for row in rows if not row["unmatched"] and row["overlap"] >= 0.5 and row["texts"]["baseline"] is not None]
    for name, layers in combos.items():
        exact = 0
        gained = []
        lost = []
        harmed = 0
        improved = 0
        for row in rows:
            if row["unmatched"]:
                continue
            baseline = row["texts"]["baseline"]
            edited = baseline
            if baseline is not None and layers:
                from multilabel.post_ocr import apply_layers

                edited = apply_layers(baseline, row.get("readers") or [], layers)
            hit = bool(row["overlap"] >= 0.5 and edited == row["gold"])
            base_hit = bool(row["overlap"] >= 0.5 and baseline == row["gold"])
            exact += int(hit)
            if hit and not base_hit:
                gained.append(
                    {
                        "document_id": row["document_id"],
                        "page": row["page"],
                        "field": row["field"],
                        "baseline": baseline,
                        "corrected": edited,
                        "gold": row["gold"],
                    }
                )
            if base_hit and not hit:
                lost.append(
                    {
                        "document_id": row["document_id"],
                        "page": row["page"],
                        "field": row["field"],
                        "baseline": baseline,
                        "corrected": edited,
                    }
                )
            if baseline is not None and edited is not None and layers:
                for outcome in word_change_outcomes(baseline, edited, row["gold"] or ""):
                    if outcome == "harmed":
                        harmed += 1
                    elif outcome == "improved":
                        improved += 1
        report[name] = {
            "exact": exact,
            "gained": gained,
            "lost": lost,
            "edits_closer": improved,
            "edits_farther": harmed,
        }
    # Second pass over the same rows. A deterministic stack returns the same strings.
    again = []
    for row in located_rows:
        again.append(stack_correction(row["texts"]["baseline"], row.get("readers") or []))
    first = [stack_correction(row["texts"]["baseline"], row.get("readers") or []) for row in located_rows]
    report["deterministic_repeat"] = first == again
    report["located"] = len(located_rows)
    return report


def render(report: dict) -> str:
    lines = [
        "# Remaining address extraction errors",
        "",
        "Labelled validation pages only. The frozen checkpoint, the 0.50 threshold,",
        "line-block-v2, and the correction stack were not changed.",
        "",
        "## Root-cause distribution",
        "",
        "Primary cause of each gold address that is not an exact raw match.",
        "A field can carry more than one issue. The second table counts every issue present.",
        "",
        "| Field | Exact | missing_ocr | ocr_character | threshold | line_grouping | unwanted_nearby | word_order |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    causes = report["causes"]
    for field, label in (
        ("vendor_address", "Vendor"),
        ("customer_billing_address", "Customer"),
    ):
        primary = causes["primary"][field]
        lines.append(
            "| {label} | {exact}/{gold} | {missing} | {ocr} | {threshold} | {grouping} | {nearby} | {order} |".format(
                label=label,
                exact=causes["exact"].get(field, 0),
                gold=causes["gold"].get(field, 0),
                missing=primary.get("missing_ocr", 0),
                ocr=primary.get("ocr_character", 0),
                threshold=primary.get("threshold", 0),
                grouping=primary.get("line_grouping", 0),
                nearby=primary.get("unwanted_nearby", 0),
                order=primary.get("word_order", 0),
            )
        )
    lines.extend(["", "Issues present on the same mismatch, including the primary cause:", ""])
    for field, label in (
        ("vendor_address", "Vendor"),
        ("customer_billing_address", "Customer"),
    ):
        issues = causes["issues_present"][field]
        pieces = ", ".join(f"{name} {count}" for name, count in sorted(issues.items()))
        lines.append(f"- {label}: {pieces}")
    patterns = report["patterns"]
    lines.extend(["", "## Low-confidence official tokens", ""])
    lines.append(
        "An official token is an OCR word inside the gold box whose letters and digits match an official word, with address probability below 0.35."
    )
    lines.append("")
    for field, label in (
        ("vendor_address", "Vendor"),
        ("customer_billing_address", "Customer"),
    ):
        item = patterns[field]
        lines.append(
            f"{label}: {item['count']} tokens. Median box height {item['median_height']} against {item['base_median_height']} for every official in-box token. Right-column share {item['right_column_share']:.3f} against a base share of {item['base_right_column_share']:.3f}."
        )
    lines.extend(
        [
            "",
            "| Field | Role | Low-confidence tokens | Share of the low-confidence set | Share of all official in-box tokens |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for field, label in (
        ("vendor_address", "Vendor"),
        ("customer_billing_address", "Customer"),
    ):
        for role, item in sorted(patterns[field]["roles"].items()):
            lines.append(
                "| {label} | {role} | {count} | {low:.3f} | {base:.3f} |".format(
                    label=label,
                    role=role,
                    count=item["low_confidence"],
                    low=item["low_share"] or 0,
                    base=item["base_share"] or 0,
                )
            )
    lines.extend(["", "Predicted label is the highest of the nine field probabilities.", ""])
    for field, label in (
        ("vendor_address", "Vendor"),
        ("customer_billing_address", "Customer"),
    ):
        labels = patterns[field]["predicted_labels"]
        if not labels or labels == {"unavailable": patterns[field]["count"]}:
            lines.append(f"- {label}: label cache not attached yet.")
            continue
        pieces = ", ".join(f"{name} {count}" for name, count in sorted(labels.items(), key=lambda item: -item[1]))
        lines.append(f"- {label}: {pieces}")
    lines.extend(["", "### Token table", ""])
    lines.append("| Document | Field | OCR | Role | Address probability | Predicted label | Box | Right column | Near heading |")
    lines.append("| --- | --- | --- | --- | ---: | --- | --- | --- | --- |")
    for row in report["tokens"]:
        lines.append(
            "| `{doc}` p{page} | {field} | {ocr} | {role} | {prob:.3f} | {label} | {box} | {column} | {heading} |".format(
                doc=row["document_id"][:12],
                page=row["page"],
                field="vendor" if row["field"] == "vendor_address" else "customer",
                ocr=row["ocr_text"].replace("|", "/"),
                role=row["role"],
                prob=row["address_probability"],
                label=row.get("predicted_label", "—"),
                box=",".join(str(value) for value in row["box"]),
                column="yes" if row["right_column"] else "no",
                heading="yes" if row["near_heading"] else "no",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    from multilabel.checkpoint import assert_diagnostic_destination

    assert_diagnostic_destination(OUTPUT)
    pages = load_pages()
    rows = low_confidence_tokens(pages)
    attach_labels(rows)
    baseline = height_baseline(pages)
    report = {
        "pages": len(pages),
        "production_changed": False,
        "causes": audit_causes(),
        "patterns": pattern_summary(rows, baseline),
        "tokens": rows,
        "named_ocr_cases": named_ocr_cases(pages),
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "tokens",
        sum(row["field"] == "vendor_address" for row in rows),
        sum(row["field"] == "customer_billing_address" for row in rows),
        "labels",
        sum("predicted_label" in row for row in rows),
    )
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
