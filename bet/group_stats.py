from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from .math_eval import is_correct
from .parsing import get_text, think_token_proxy
from .schemas import GroupProfile


def prompt_key(prompt: Any) -> str:
    return get_text(prompt).strip()


def compute_group_profiles(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answers: Sequence[Any],
    *,
    max_completion_tokens: float,
    efficient_cost_percentile: float = 0.30,
    tokenizer: Any = None,
) -> Dict[str, GroupProfile]:
    """Group solvability, efficient solution cost, and budget target per query."""
    groups: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)
    for p, c, a in zip(prompts, completions, answers):
        groups[prompt_key(p)].append((c, a))

    profiles: Dict[str, GroupProfile] = {}
    for key, items in groups.items():
        correct_flags = [is_correct(c, a) for c, a in items]
        lengths = [think_token_proxy(c, tokenizer) for c, _ in items]
        correct_lengths = [l for l, ok in zip(lengths, correct_flags) if ok]
        n = len(items)
        num_correct = sum(correct_flags)
        s_hat = num_correct / n if n else 0.0
        if correct_lengths:
            sorted_lengths = sorted(correct_lengths)
            k = max(1, math.ceil(len(sorted_lengths) * efficient_cost_percentile))
            c_star = sum(sorted_lengths[:k]) / k
            b_star = min(1.0, c_star / max(1.0, max_completion_tokens))
        else:
            c_star = 0.0
            b_star = 0.0
        profiles[key] = GroupProfile(
            prompt_key=key,
            n=n,
            num_correct=num_correct,
            solvability=s_hat,
            efficient_cost=c_star,
            budget_target=b_star,
            correct_lengths=correct_lengths,
        )
    return profiles
