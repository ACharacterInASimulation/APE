"""Small QA metrics used by the scratchpad/APE eval scripts."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def best_subspan_em(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    for answer in answers or [""]:
        gold = normalize_answer(answer)
        if gold and gold in pred:
            return 1.0
    return float(not pred and not answers)


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not pred_tokens or not answer_tokens:
        return float(pred_tokens == answer_tokens)
    common = Counter(pred_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def score_qa(prediction: str, answers: list[str]) -> dict[str, float]:
    answers = answers or [""]
    first_line = prediction.splitlines()[0] if prediction else prediction
    return {
        "exact_match": max(exact_match(first_line, answer) for answer in answers),
        "token_f1": max(token_f1(first_line, answer) for answer in answers),
        "best_subspan_em": best_subspan_em(prediction, answers),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }
