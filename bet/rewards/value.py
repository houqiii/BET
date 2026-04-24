from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from ..group_stats import prompt_key
from ..math_eval import is_correct
from ..parsing import parse_response, think_token_proxy


def failure_penalty(length: float, max_completion_tokens: float, solvability: float) -> float:
    ratio = min(1.0, length / max(1.0, max_completion_tokens))
    if solvability <= 0.0:
        return 0.60 + 0.20 * ratio
    if solvability < 0.5:
        return 0.35 + 0.05 * ratio
    return 0.50 + 0.20 * ratio


def score_value(
    prompt: Any,
    completion: Any,
    answer: Any,
    profiles: Mapping[str, Any],
    *,
    delta: float,
    lambda_abstain: float,
    max_completion_tokens: float,
) -> float:
    profile = profiles[prompt_key(prompt)]
    parsed = parse_response(completion)
    if is_correct(completion, answer):
        return 1.0
    if parsed.is_fold:
        return delta if profile.num_correct == 0 else -lambda_abstain
    length = think_token_proxy(completion)
    return -failure_penalty(length, max_completion_tokens, profile.solvability)


def reward_value(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answer: Sequence[Any],
    *,
    profiles: Mapping[str, Any],
    delta: float = 0.10,
    lambda_abstain: float = 0.80,
    max_completion_tokens: float = 16384,
    **kwargs: Any,
) -> List[float]:
    return [
        score_value(p, c, a, profiles, delta=delta, lambda_abstain=lambda_abstain, max_completion_tokens=max_completion_tokens)
        for p, c, a in zip(prompts, completions, answer)
    ]
