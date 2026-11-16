"""LayoutLMv3 encoder plus nine independent field logits."""

from __future__ import annotations

import torch
from torch import nn

from multilabel.loss import masked_bce_with_logits


class MultiLabelLayoutLM(nn.Module):
    def __init__(self, encoder: nn.Module, num_labels: int = 9):
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
        )
        # Image patch tokens are appended after the text. Labels exist only for text.
        text_length = input_ids.size(1)
        logits = self.classifier(outputs.last_hidden_state[:, :text_length])
        if labels is None or loss_mask is None:
            return logits, None
        return logits, masked_bce_with_logits(logits, labels, loss_mask)
