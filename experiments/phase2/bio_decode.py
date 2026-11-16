"""BIO inspection helpers and a decoder that can recover a missing B tag.

Recovery is a reading rule for tags the model already emitted. It does not
put probability back on the B classes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedSpan:
    label: str
    start: int
    end: int
    recovered: bool

    @property
    def word_indexes(self) -> range:
        return range(self.start, self.end)


def decode_bio_labels(labels: list[str]) -> list[DecodedSpan]:
    """Turn word tags into spans.

    B starts a new entity.
    I continues an open entity of the same type.
    I after O, or after a different type, starts a recovered entity.
    O closes the current entity.
    """
    spans: list[DecodedSpan] = []
    current: tuple[int, str, bool] | None = None

    def close(end: int) -> None:
        nonlocal current
        if current is None:
            return
        start, kind, recovered = current
        if end > start:
            spans.append(DecodedSpan(kind, start, end, recovered))
        current = None

    for index, label in enumerate(labels):
        tag, kind = _split_label(label)
        if tag == "B" and kind:
            close(index)
            current = (index, kind, False)
            continue
        if tag == "I" and kind:
            if current is not None and current[1] == kind:
                continue
            close(index)
            current = (index, kind, True)
            continue
        close(index)
    close(len(labels))
    return spans


def _split_label(label: str) -> tuple[str, str]:
    if not label or label == "O" or "-" not in label:
        return "O", ""
    tag, kind = label.split("-", 1)
    if tag not in {"B", "I"} or not kind:
        return "O", ""
    return tag, kind


def count_bio(labels: list[str]) -> dict[str, int]:
    counts = {"B": 0, "I": 0, "O": 0}
    for label in labels:
        tag, _kind = _split_label(label)
        counts[tag] = counts.get(tag, 0) + 1
    return counts


def count_bio_by_field(labels: list[str], field_names: list[str]) -> dict[str, dict[str, int]]:
    table = {name: {"B": 0, "I": 0} for name in field_names}
    table["O"] = {"O": 0}
    for label in labels:
        tag, kind = _split_label(label)
        if tag == "O":
            table["O"]["O"] += 1
            continue
        table.setdefault(kind, {"B": 0, "I": 0})
        table[kind][tag] = table[kind].get(tag, 0) + 1
    return table


def mapping_errors(id2label: dict, label_list: list[str]) -> list[str]:
    """Return mismatches between a checkpoint map and the training label order."""
    errors: list[str] = []
    if len(id2label) != len(label_list):
        errors.append(f"map has {len(id2label)} labels, list has {len(label_list)}")
    for index, name in enumerate(label_list):
        observed = id2label.get(index, id2label.get(str(index)))
        if observed != name:
            errors.append(f"id {index} is {observed!r}, expected {name!r}")
    reverse = {}
    for key, value in id2label.items():
        reverse.setdefault(value, []).append(key)
    for name, keys in reverse.items():
        if len(keys) > 1:
            errors.append(f"{name} is assigned to ids {keys}")
    return errors


def first_subword_agreement(
    word_labels: list[str],
    word_ids: list[int | None],
    token_labels: list[str],
) -> dict[str, int]:
    """Check that each word's first subword keeps its BIO tag.

    Later subwords may be ignored. Words absent from word_ids were truncated.
    """
    seen: set[int] = set()
    kept_b = 0
    changed_b = 0
    kept_other = 0
    changed_other = 0
    for token_label, word_id in zip(token_labels, word_ids):
        if word_id is None or word_id in seen or word_id >= len(word_labels):
            continue
        seen.add(word_id)
        gold = word_labels[word_id]
        if gold.startswith("B-"):
            if token_label == gold:
                kept_b += 1
            else:
                changed_b += 1
        elif token_label == gold:
            kept_other += 1
        else:
            changed_other += 1
    b_words = sum(1 for label in word_labels if label.startswith("B-"))
    return {
        "b_words": b_words,
        "b_kept_on_first_subword": kept_b,
        "b_changed_on_first_subword": changed_b,
        "b_truncated": b_words - kept_b - changed_b,
        "other_kept_on_first_subword": kept_other,
        "other_changed_on_first_subword": changed_other,
    }
