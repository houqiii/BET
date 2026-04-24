from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from ..group_stats import prompt_key
from ..math_eval import is_correct
from ..parsing import think_token_proxy


def score_efficiency(
    prompt: Any,
    completion: Any,
    answer: Any,
    profiles: Mapping[str, Any],
    *,
    beta: float,
    tau: float,
) -> float:
    profile = profiles[prompt_key(prompt)]
    if profile.solvability <= tau or not is_correct(completion, answer):
        return 0.0
    if profile.efficient_cost <= 0:
        return 0.0
    length = think_token_proxy(completion)
    return beta * max(0.0, 1.0 - length / profile.efficient_cost)


def reward_efficiency(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answer: Sequence[Any],
    *,
    profiles: Mapping[str, Any],
    beta: float = 0.30,
    tau: float = 0.25,
    **kwargs: Any,
) -> List[float]:
    return [score_efficiency(p, c, a, profiles, beta=beta, tau=tau) for p, c, a in zip(prompts, completions, answer)]
