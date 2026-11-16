"""Put word targets on the first subword. Other tokens are masked out of the loss."""

from __future__ import annotations


def align_word_targets(
    word_targets: list[list[int]],
    word_ids: list[int | None],
    num_labels: int,
) -> dict:
    """Return token targets, a loss mask, and the word index of each token.

    Continuation subwords and special tokens get mask 0. Their target values
    are zeros and must not affect the loss.
    """
    length = len(word_ids)
    labels = [[0.0] * num_labels for _ in range(length)]
    loss_mask = [0] * length
    token_word = [-1] * length
    seen: set[int] = set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen or word_id >= len(word_targets):
            continue
        seen.add(word_id)
        labels[position] = [float(value) for value in word_targets[word_id]]
        loss_mask[position] = 1
        token_word[position] = word_id
    truncated = [index for index in range(len(word_targets)) if index not in seen]
    return {
        "labels": labels,
        "loss_mask": loss_mask,
        "token_word": token_word,
        "truncated_words": truncated,
    }


def word_probabilities(
    token_probabilities: list[list[float]],
    word_ids: list[int | None],
    word_count: int,
) -> list[list[float]]:
    """Read each word's probabilities from its first subword."""
    width = len(token_probabilities[0]) if token_probabilities else 0
    words = [[0.0] * width for _ in range(word_count)]
    seen: set[int] = set()
    for position, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen or word_id >= word_count:
            continue
        if position >= len(token_probabilities):
            continue
        seen.add(word_id)
        words[word_id] = [float(value) for value in token_probabilities[position]]
    return words
