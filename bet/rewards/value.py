"""R_VAL: Solution value with abstention gate (Eq. 4)."""
from __future__ import annotations

from typing import Any, List, Mapping, Sequence

from ..group_stats import prompt_key
from ..math_eval import is_correct
from ..parsing import parse_response, think_token_proxy


def score_value(
    prompt: Any,
    completion: Any,
    answer: Any,
    profiles: Mapping[str, Any],
    *,
    delta: float,
    lambda_abstain: float,
    alpha_fail: float,
    max_completion_tokens: float,
) -> float:
    """Per-trajectory R_VAL as defined in Eq. 4.

    - Correct solution:  +1
    - Incorrect attempt: -phi(c(y)), where phi(c) = alpha_fail * c / L_max
    - Abstention:        +delta   if s_hat(x) < epsilon_abs,
                         -lambda  if s_hat(x) >= epsilon_abs.
    """
    profile = profiles[prompt_key(prompt)]
    parsed = parse_response(completion)
    if is_correct(completion, answer):
        return 1.0
    if parsed.is_fold:
        # epsilon_abs = 1/K, equivalent to num_correct == 0
        return delta if profile.num_correct == 0 else -lambda_abstain
    # phi(c) = alpha_fail * c(y) / L_max
    length = think_token_proxy(completion)
    ratio = min(1.0, length / max(1.0, max_completion_tokens))
    return -alpha_fail * ratio


def reward_value(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answer: Sequence[Any],
    *,
    profiles: Mapping[str, Any],
    delta: float = 0.10,
    lambda_abstain: float = 0.80,
    alpha_fail: float = 0.20,
    max_completion_tokens: float = 16384,
    **kwargs: Any,
) -> List[float]:
    return [
        score_value(
            p, c, a, profiles,
            delta=delta,
            lambda_abstain=lambda_abstain,
            alpha_fail=alpha_fail,
            max_completion_tokens=max_completion_tokens,
        )
        for p, c, a in zip(prompts, completions, answer)
    ]
