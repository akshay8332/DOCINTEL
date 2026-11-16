"""Keep every overlapping field, and score tokens with independent sigmoids.

The softmax labeler keeps only the smallest box around a word. This module
keeps every field whose box contains the word center. The head is a prototype:
nothing here updates the saved checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _contains(box: list[float], cx: float, cy: float) -> bool:
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def containing_fields(
    words: list[tuple[str, list[float]]],
    fields: list[tuple[str, list[float]]],
    *,
    allowed: set[str] | None = None,
) -> list[list[int]]:
    """Field indexes whose boxes contain each word center, smallest first."""
    keep = allowed
    assigned: list[list[int]] = []
    for _text, box in words:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        hits = [
            index
            for index, (name, field_box) in enumerate(fields)
            if (keep is None or name in keep) and _contains(field_box, cx, cy)
        ]
        hits.sort(key=lambda index: _area(fields[index][1]))
        assigned.append(hits)
    return assigned


def multilabel_matrix(
    words: list[tuple[str, list[float]]],
    fields: list[tuple[str, list[float]]],
    field_names: list[str],
) -> list[list[int]]:
    """One multi-hot row per word. Several field types may be 1 together."""
    name_to_index = {name: index for index, name in enumerate(field_names)}
    rows = []
    for hits in containing_fields(words, fields, allowed=set(field_names)):
        row = [0] * len(field_names)
        for field_index in hits:
            name = fields[field_index][0]
            column = name_to_index.get(name)
            if column is not None:
                row[column] = 1
        rows.append(row)
    return rows


def exclusive_name(
    words: list[tuple[str, list[float]]],
    fields: list[tuple[str, list[float]]],
    *,
    allowed: set[str] | None = None,
) -> list[str | None]:
    """The single field the current softmax labeler would keep."""
    chosen = []
    for hits in containing_fields(words, fields, allowed=allowed):
        chosen.append(fields[hits[0]][0] if hits else None)
    return chosen


def annotation_retention(
    words: list[tuple[str, list[float]]],
    fields: list[tuple[str, list[float]]],
    *,
    allowed: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Count annotations that still have at least one labeled word.

    ``multi`` keeps every containing field. ``exclusive`` keeps the smallest.
    """
    hits = containing_fields(words, fields, allowed=allowed)
    multi_kept = {index for word_hits in hits for index in word_hits}
    exclusive_kept = {word_hits[0] for word_hits in hits if word_hits}
    per_field: dict[str, dict[str, int]] = {}
    for index, (name, _box) in enumerate(fields):
        if allowed is not None and name not in allowed:
            continue
        row = per_field.setdefault(name, {"annotations": 0, "exclusive": 0, "multi": 0})
        row["annotations"] += 1
        row["exclusive"] += int(index in exclusive_kept)
        row["multi"] += int(index in multi_kept)
    return per_field


class IndependentSigmoidHead(nn.Module):
    """One sigmoid per field type. Loss is BCE-with-logits, with no class weights."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.classifier(hidden_states)
        if labels is None:
            return logits, None
        return logits, multilabel_bce(logits, labels, attention_mask)


def multilabel_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean BCE over active tokens. ``labels`` are 0/1 floats."""
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    if attention_mask is None:
        return loss.mean()
    mask = attention_mask.unsqueeze(-1).to(loss.dtype)
    active = mask.sum() * logits.size(-1)
    return (loss * mask).sum() / active.clamp(min=1)


def field_scores(
    gold: list[list[int]],
    prediction: list[list[int]],
    field_names: list[str],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Per-field precision, recall and F1 for multi-hot word decisions."""
    scores = {}
    for column, name in enumerate(field_names):
        tp = fp = fn = 0
        for gold_row, pred_row in zip(gold, prediction):
            truth = int(gold_row[column]) == 1
            guessed = (
                float(pred_row[column]) >= threshold
                if not isinstance(pred_row[column], int)
                else int(pred_row[column]) == 1
            )
            if truth and guessed:
                tp += 1
            elif guessed:
                fp += 1
            elif truth:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        scores[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return scores
