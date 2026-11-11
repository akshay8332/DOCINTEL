"""Entity-level precision / recall / F1 for FUNSD BIO tags."""

from __future__ import annotations

from typing import Any, Sequence


def seqeval_scores(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Micro / macro entity F1 via seqeval (same convention as LayoutLM papers)."""
    from seqeval.metrics import (
        f1_score,
        precision_score,
        recall_score,
        classification_report,
    )

    return {
        "precision_micro": float(precision_score(y_true, y_pred)),
        "recall_micro": float(recall_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred),
    }


def entity_f1_report(
    y_true: Sequence[Sequence[str]],
    y_pred: Sequence[Sequence[str]],
) -> dict[str, Any]:
    scores = seqeval_scores(y_true, y_pred)
    return {
        "f1_micro": scores["f1_micro"],
        "f1_macro": scores["f1_macro"],
        "precision_micro": scores["precision_micro"],
        "recall_micro": scores["recall_micro"],
        "detail": scores["report"],
    }


def align_predictions(
    predictions: Sequence[Sequence[int]],
    label_ids: Sequence[Sequence[int]],
    id2label: dict[int, str],
    ignore_index: int = -100,
) -> tuple[list[list[str]], list[list[str]]]:
    """Convert model logit argmax + label ids into seqeval BIO string lists."""
    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []
    for pred_seq, lab_seq in zip(predictions, label_ids):
        t_row: list[str] = []
        p_row: list[str] = []
        for p, lab in zip(pred_seq, lab_seq):
            if lab == ignore_index:
                continue
            t_row.append(id2label[int(lab)])
            p_row.append(id2label[int(p)])
        true_labels.append(t_row)
        pred_labels.append(p_row)
    return true_labels, pred_labels
