"""R_CAL: Solvability calibration (Eq. 6)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..group_stats import prompt_key
from ..parsing import parse_response


def asymmetric_budget_loss(
    b_pred: float, b_star: float, mu: float,
) -> float:
    """Asymmetric budget loss ell(b_hat, b*) as defined in Table 7.

    ell(b_hat, b*) = mu * |b_hat - b*|  if b_hat < b*  (underestimation)
                     |b_hat - b*|        if b_hat >= b* (overestimation)

    mu > 1 makes under-budgeting costlier than over-budgeting.
    """
    gap = abs(b_pred - b_star)
    if b_pred < b_star:
        return mu * gap
    return gap


def score_calibration(
    prompt: Any,
    completion: Any,
    profiles: Mapping[str, Any],
    *,
    gamma_s: float,
    gamma_b: float,
    gamma_s_unsolvable: float,
    gamma_b_unsolvable: float,
    mu: float,
) -> Tuple[float, Dict[str, float]]:
    """Per-trajectory R_CAL as defined in Eq. 6.

    Solvable regime (s_hat(x) >= epsilon_abs):
      R_CAL = -gamma_s * |s_pred - s_hat(x)| - gamma_b * ell(b_pred, b*(x))

    Zero-return regime (s_hat(x) < epsilon_abs):
      R_CAL = -gamma'_s * s_pred - gamma'_b * b_pred

    where b*(x) = c_hat*(x) / L_max is the budget target (Table 7).
    """
    profile = profiles[prompt_key(prompt)]
    parsed = parse_response(completion)
    s_pred = 0.5 if parsed.solvability_pred is None else parsed.solvability_pred
    b_pred = 0.5 if parsed.budget is None else parsed.budget

    if profile.num_correct == 0:
        # Zero-return regime: push s_pred → 0, b_pred → 0
        score = -gamma_s_unsolvable * s_pred - gamma_b_unsolvable * b_pred
        return score, {
            "s_err": s_pred,
            "b_err": b_pred,
            "target_s": 0.0,
            "target_b": 0.0,
        }

    # Solvable regime: calibrate against group profile
    s_err = abs(s_pred - profile.solvability)
    b_loss = asymmetric_budget_loss(b_pred, profile.budget_target, mu)
    score = -gamma_s * s_err - gamma_b * b_loss
    return score, {
        "s_err": s_err,
        "b_err": abs(b_pred - profile.budget_target),
        "target_s": profile.solvability,
        "target_b": profile.budget_target,
    }


def reward_calibration(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    *,
    profiles: Mapping[str, Any],
    gamma_s: float = 0.10,
    gamma_b: float = 0.20,
    gamma_s_unsolvable: float = 0.20,
    gamma_b_unsolvable: float = 0.10,
    mu: float = 2.0,
    **kwargs: Any,
) -> List[float]:
    return [
        score_calibration(
            p, c, profiles,
            gamma_s=gamma_s,
            gamma_b=gamma_b,
            gamma_s_unsolvable=gamma_s_unsolvable,
            gamma_b_unsolvable=gamma_b_unsolvable,
            mu=mu,
        )[0]
        for p, c in zip(prompts, completions)
    ]
