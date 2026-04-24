from __future__ import annotations

from typing import Any, List, Sequence

from ..parsing import inspect_format


def score_format(text: Any) -> float:
    info = inspect_format(text)
    if info["format_ok"]:
        return 0.15
    score = 0.0
    if not info["starts_with_predict"]:
        score -= 0.15
    if not info["predict_count_ok"]:
        score -= 0.15
    if not info["predict_parse_ok"]:
        score -= 0.15
    if not info["think_count_ok"]:
        score -= 0.45
    if not info["think_parse_ok"]:
        score -= 0.30
    if not info["boxed_ok"]:
        score -= 1.20
    if not info["boxed_count_ok"]:
        score -= 0.35
    if not info["order_ok"]:
        score -= 0.25
    return max(score, -2.0)


def reward_format(prompts: Sequence[Any], completions: Sequence[Any], **kwargs: Any) -> List[float]:
    return [score_format(c) for c in completions]
