from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .math_eval import is_correct
from .parsing import get_text, think_token_proxy
from .schemas import GroupProfile


def prompt_key(prompt: Any) -> str:
    return get_text(prompt).strip()


def budget_from_length(length: float, max_completion_tokens: float) -> float:
    raw = float(length) / max(1.0, float(max_completion_tokens))
    # Decile target used for stable calibration rather than an overly precise scalar.
    return min(1.0, max(0.1, math.ceil(raw * 10.0 - 1e-8) / 10.0))


def difficulty_target(s_hat: float, b_star: float, zero_return: bool) -> float:
    if zero_return:
        return 1.0
    return min(0.9, max(0.0, 0.35 * (1.0 - s_hat) + 0.65 * b_star))


def compute_group_profiles(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answers: Sequence[Any],
    *,
    max_completion_tokens: float,
    efficient_cost_percentile: float = 0.30,
) -> Dict[str, GroupProfile]:
    groups: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)
    for p, c, a in zip(prompts, completions, answers):
        groups[prompt_key(p)].append((c, a))

    profiles: Dict[str, GroupProfile] = {}
    for key, items in groups.items():
        correct_flags = [is_correct(c, a) for c, a in items]
        lengths = [think_token_proxy(c) for c, _ in items]
        correct_lengths = [l for l, ok in zip(lengths, correct_flags) if ok]
        n = len(items)
        num_correct = sum(correct_flags)
        s_hat = num_correct / n if n else 0.0
        if correct_lengths:
            sorted_lengths = sorted(correct_lengths)
            k = max(1, math.ceil(len(sorted_lengths) * efficient_cost_percentile))
            c_star = sum(sorted_lengths[:k]) / k
            b_star = budget_from_length(c_star, max_completion_tokens)
        else:
            c_star = 0.0
            b_star = 0.0
        d_star = difficulty_target(s_hat, b_star, zero_return=(num_correct == 0))
        profiles[key] = GroupProfile(
            prompt_key=key,
            n=n,
            num_correct=num_correct,
            solvability=s_hat,
            efficient_cost=c_star,
            budget_target=b_star,
            difficulty_target=d_star,
            correct_lengths=correct_lengths,
        )
    return profiles
