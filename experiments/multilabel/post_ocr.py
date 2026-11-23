"""Selective edits to a prediction string. The checkpoint and DocILE files stay unchanged."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from multilabel.address_text_audit import norm_token, tokens
from multilabel.ocr_text import correct_common_ocr

_CLO = re.compile(r"^clo$", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_PO_DOTS = re.compile(r"\bP\.O\.\.", re.IGNORECASE)


def wording_tokens(text: str) -> list[str]:
    return [token for token in (norm_token(part) for part in tokens(text or "")) if token]


def same_wording(left: str, right: str) -> bool:
    return wording_tokens(left) == wording_tokens(right)


def normalize_prediction(text: str) -> str:
    """Tighten spaces and punctuation without changing letters or digits."""
    lines = []
    for line in (text or "").split("\n"):
        cleaned = _MULTI_SPACE.sub(" ", line.strip())
        cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)


def correct_prediction(text: str) -> str:
    """Apply the known character fixes, then the spacing normalization."""
    corrected = correct_common_ocr(text or "")
    corrected = _PO_DOTS.sub("P.O.", corrected)
    corrected = "\n".join(
        " ".join("c/o" if _CLO.fullmatch(token) else token for token in line.split(" "))
        for line in corrected.split("\n")
    )
    return normalize_prediction(corrected)


def refine_boundary(text: str) -> str:
    """Drop a repeated line, or a lone leading digit, and leave company lines in place."""
    lines = [line for line in (text or "").split("\n")]
    kept: list[str] = []
    for line in lines:
        if kept and line == kept[-1]:
            continue
        kept.append(line)
    if kept:
        parts = kept[0].split(" ")
        if len(parts) >= 2 and len(parts[0]) == 1 and parts[0].isdigit():
            kept[0] = " ".join(parts[1:])
    return "\n".join(kept)


def _content(text: str) -> list[tuple[str, str]]:
    return [(raw, norm) for raw in tokens(text) if (norm := norm_token(raw))]


def _levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _spans(source: list[str], other: list[str]) -> dict[int, tuple]:
    mapping: dict[int, tuple] = {}
    matcher = SequenceMatcher(a=source, b=other, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(left_end - left_start):
                mapping[left_start + offset] = ("equal", right_start + offset)
        elif tag == "replace" and left_end - left_start == 1:
            mapping[left_start] = ("replace", right_start, right_end)
        else:
            for index in range(left_start, left_end):
                mapping[index] = ("keep",)
    return mapping


def selective_fallback(prediction: str, reader_texts: list[str]) -> str:
    """Replace one DocILE token only when both other readers agree on a near alternative.

    Tokens are not deleted. A token may be split when both readers agree and the
    pieces still contain the same letters and digits.
    """
    readers = [text for text in reader_texts if text]
    if len(readers) < 2 or not prediction:
        return prediction or ""
    original = _content(prediction)
    if not original:
        return prediction
    alternatives = [_content(text) for text in readers[:2]]
    if any(not pairs for pairs in alternatives):
        return prediction
    maps = [_spans([norm for _, norm in original], [norm for _, norm in pairs]) for pairs in alternatives]
    replacements: dict[int, list[str]] = {}
    for index, (_raw, norm) in enumerate(original):
        spans = [item.get(index, ("keep",)) for item in maps]
        if any(span[0] != "replace" for span in spans):
            continue
        pieces = []
        agreed = True
        norms: list[str] | None = None
        for span, pairs in zip(spans, alternatives):
            start, end = span[1], span[2]
            chunk = pairs[start:end]
            chunk_norms = [item[1] for item in chunk]
            if norms is None:
                norms = chunk_norms
            elif chunk_norms != norms:
                agreed = False
                break
            pieces.append([item[0] for item in chunk])
        if not agreed or not norms:
            continue
        if len(norms) == 1 and 0 < _levenshtein(norm, norms[0]) <= 2:
            replacements[index] = [pieces[0][0]]
        elif len(norms) == 2 and "".join(norms) == norm:
            replacements[index] = pieces[0]
    if not replacements:
        return prediction
    content_index = 0
    lines = []
    for line in prediction.split("\n"):
        rebuilt = []
        for raw in line.split(" "):
            if raw == "":
                continue
            if norm_token(raw):
                rebuilt.extend(replacements.get(content_index, [raw]))
                content_index += 1
            else:
                rebuilt.append(raw)
        lines.append(" ".join(rebuilt))
    return "\n".join(lines)


def apply_layers(prediction: str, reader_texts: list[str], layers: tuple[str, ...]) -> str:
    """Apply named layers in the given order. The stack order is correct, boundary, fallback."""
    text = prediction or ""
    for layer in layers:
        if layer == "correct":
            text = correct_prediction(text)
        elif layer == "boundary":
            text = refine_boundary(text)
        elif layer == "fallback":
            text = selective_fallback(text, reader_texts)
        else:
            raise KeyError(layer)
    return text


def stack_correction(prediction: str, reader_texts: list[str]) -> str:
    """Character fixes, then the narrow boundary rule, then an agreed reader substitution."""
    return apply_layers(prediction, reader_texts, ("correct", "boundary", "fallback"))


def _closest(span: list[str], official: list[str]) -> int:
    best = sequence_distance(span, official)
    for start in range(len(official) + 1):
        for end in range(start, min(len(official), start + len(span) + 2) + 1):
            best = min(best, sequence_distance(span, official[start:end]))
    return best


def wording_edit_count(before: str, after: str) -> int:
    source = wording_tokens(before)
    edited = wording_tokens(after)
    matcher = SequenceMatcher(a=source, b=edited, autojunk=False)
    return sum(1 for tag, *_rest in matcher.get_opcodes() if tag != "equal")


def word_change_outcomes(before: str, after: str, gold: str) -> list[str]:
    """Label each wording-token edit as improved, harmed, or neutral against the official text."""
    source = wording_tokens(before)
    edited = wording_tokens(after)
    official = wording_tokens(gold)
    outcomes = []
    matcher = SequenceMatcher(a=source, b=edited, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        old = source[left_start:left_end]
        new = edited[right_start:right_end]
        before_distance = _closest(old, official)
        after_distance = _closest(new, official)
        if after_distance < before_distance:
            outcomes.append("improved")
        elif after_distance > before_distance:
            outcomes.append("harmed")
        else:
            outcomes.append("neutral")
    return outcomes


def sequence_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, start=1):
        current = [index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]
