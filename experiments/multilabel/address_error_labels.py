"""Cache all nine epoch-3 field probabilities for the labelled validation pages.

The checkpoint is read only. The production threshold and extraction service
are not changed. A second forward pass of the first page checks that the
probabilities repeat.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "experiments"))

OUTPUT = ROOT / "artifacts" / "diagnostics" / "address-error-epoch3-labels.jsonl"


def main() -> int:
    import torch

    from docintel.data.docile import load_docile_pages
    from multilabel.align import word_probabilities
    from multilabel.checkpoint import assert_diagnostic_destination
    from multilabel.data import encode_document
    from multilabel.evaluate import load_multilabel

    assert_diagnostic_destination(OUTPUT)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = ROOT / "artifacts" / "checkpoints" / "epoch-3"
    print(f"Reading checkpoint {checkpoint}", flush=True)
    model, processor, _payload = load_multilabel(checkpoint, device)
    docs = load_docile_pages(
        ROOT / "data" / "docile",
        "val",
        limit=100,
        image_dir=ROOT / "data" / "docile" / "page_images",
    )
    print(f"Loaded pages {len(docs)}", flush=True)

    def scores_for(doc) -> tuple[list[str], list[list[int]], list[list[float]], list[int]]:
        page = encode_document(doc, processor, max_length=512)
        batch = {
            key: page[key].unsqueeze(0).to(device)
            for key in ("input_ids", "attention_mask", "bbox", "pixel_values")
        }
        with torch.inference_mode():
            logits, _loss = model(**batch)
        probabilities = word_probabilities(
            torch.sigmoid(logits[0]).cpu().tolist(),
            page["word_ids"],
            len(page["words"]),
        )
        seen = {word_id for word_id in page["word_ids"] if word_id is not None}
        truncated = [index for index in range(len(page["words"])) if index not in seen]
        return page["words"], page["boxes"], probabilities, truncated

    first_words, first_boxes, first_scores, _truncated = scores_for(docs[0])
    repeat_words, repeat_boxes, repeat_scores, _repeat_truncated = scores_for(docs[0])
    delta = max(
        abs(left - right)
        for row, other in zip(first_scores, repeat_scores)
        for left, right in zip(row, other)
    )
    print(
        f"repeat page 0 same_text {first_words == repeat_words} same_boxes {first_boxes == repeat_boxes} max_abs_delta {delta:.3e}",
        flush=True,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for count, doc in enumerate(docs, start=1):
            words, boxes, probabilities, truncated = (
                (first_words, first_boxes, first_scores, _truncated) if count == 1 else scores_for(doc)
            )
            doc_id, page_s = doc.document_id.rsplit("-p", 1)
            handle.write(
                json.dumps(
                    {
                        "document_id": doc_id,
                        "page": int(page_s),
                        "words": [
                            {
                                "text": text,
                                "box": [int(value) for value in boxes[index]],
                                "scores": [round(float(value), 6) for value in probabilities[index]],
                                "truncated": index in set(truncated),
                            }
                            for index, text in enumerate(words)
                        ],
                    }
                )
                + "\n"
            )
            if count % 20 == 0:
                print(f"scored {count}", flush=True)
    print(f"Wrote {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
