from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..group_stats import prompt_key
from ..parsing import parse_response


def asymmetric_budget_loss(pred: float, target: float, under_weight: float, over_weight: float) -> float:
    under = max(0.0, target - pred)
    over = max(0.0, pred - target)
    return under_weight * under + over_weight * over


def score_calibration(
    prompt: Any,
    completion: Any,
    profiles: Mapping[str, Any],
    *,
    gamma_d: float,
    gamma_b_under: float,
    gamma_b_over: float,
    gamma_d_unsolvable: float,
    gamma_b_unsolvable: float,
) -> Tuple[float, Dict[str, float]]:
    profile = profiles[prompt_key(prompt)]
    parsed = parse_response(completion)
    d_pred = 0.5 if parsed.difficulty is None else parsed.difficulty
    b_pred = 0.5 if parsed.budget is None else parsed.budget

    if profile.num_correct == 0:
        d_err = abs(d_pred - 1.0)
        score = -gamma_d_unsolvable * d_err - gamma_b_unsolvable * b_pred
        return score, {"d_err": d_err, "b_err": b_pred, "target_d": 1.0, "target_b": 0.0}

    d_err = abs(d_pred - profile.difficulty_target)
    b_loss = asymmetric_budget_loss(b_pred, profile.budget_target, gamma_b_under, gamma_b_over)
    score = -gamma_d * d_err - b_loss
    return score, {"d_err": d_err, "b_err": abs(b_pred - profile.budget_target), "target_d": profile.difficulty_target, "target_b": profile.budget_target}


def reward_calibration(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    *,
    profiles: Mapping[str, Any],
    gamma_d: float = 0.10,
    gamma_b_under: float = 0.20,
    gamma_b_over: float = 0.10,
    gamma_d_unsolvable: float = 0.40,
    gamma_b_unsolvable: float = 0.40,
    **kwargs: Any,
) -> List[float]:
    return [
        score_calibration(
            p,
            c,
            profiles,
            gamma_d=gamma_d,
            gamma_b_under=gamma_b_under,
            gamma_b_over=gamma_b_over,
            gamma_d_unsolvable=gamma_d_unsolvable,
            gamma_b_unsolvable=gamma_b_unsolvable,
        )[0]
        for p, c in zip(prompts, completions)
    ]
