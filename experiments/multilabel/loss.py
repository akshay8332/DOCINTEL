"""Masked BCE-with-logits. Sigmoid is inside the loss, not applied beforehand."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Average BCE over supervised positions only.

    ``loss_mask`` is 1 for the first subword of a real OCR word and 0 for
    continuation subwords, special tokens, and padding. No class weights.
    """
    raw = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    mask = loss_mask.to(raw.dtype).unsqueeze(-1)
    denominator = (mask.sum() * logits.size(-1)).clamp(min=1)
    return (raw * mask).sum() / denominator
