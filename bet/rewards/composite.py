"""Composite BET reward R(y|x) = R_VAL + R_EFF + R_CAL (Section 3.3)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Sequence

from ..group_stats import compute_group_profiles
from ..schemas import RewardBreakdown
from .calibration import score_calibration
from .efficiency import score_efficiency
from .format import score_format
from .value import score_value


@dataclass
class BETRewardConfig:
    """Reward hyperparameters matching Table 6 (Appendix A.4)."""
    max_completion_tokens: int = 16384
    efficient_cost_percentile: float = 0.30
    # R_VAL (Eq. 4)
    delta: float = 0.10
    lambda_abstain: float = 0.80
    alpha_fail: float = 0.20
    # R_EFF (Eq. 5)
    beta: float = 0.30
    tau: float = 0.25
    # R_CAL (Eq. 6)
    gamma_s: float = 0.10
    gamma_b: float = 0.20
    gamma_s_unsolvable: float = 0.20
    gamma_b_unsolvable: float = 0.10
    mu: float = 2.0
    # Format reward
    include_format_reward: bool = True


def compute_bet_rewards(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answers: Sequence[Any],
    config: BETRewardConfig | None = None,
) -> List[RewardBreakdown]:
    cfg = config or BETRewardConfig()
    profiles = compute_group_profiles(
        prompts,
        completions,
        answers,
        max_completion_tokens=cfg.max_completion_tokens,
        efficient_cost_percentile=cfg.efficient_cost_percentile,
    )
    out: List[RewardBreakdown] = []
    for p, c, a in zip(prompts, completions, answers):
        r_val = score_value(
            p, c, a, profiles,
            delta=cfg.delta,
            lambda_abstain=cfg.lambda_abstain,
            alpha_fail=cfg.alpha_fail,
            max_completion_tokens=cfg.max_completion_tokens,
        )
        r_eff = score_efficiency(p, c, a, profiles, beta=cfg.beta, tau=cfg.tau)
        r_cal, _ = score_calibration(
            p, c, profiles,
            gamma_s=cfg.gamma_s,
            gamma_b=cfg.gamma_b,
            gamma_s_unsolvable=cfg.gamma_s_unsolvable,
            gamma_b_unsolvable=cfg.gamma_b_unsolvable,
            mu=cfg.mu,
        )
        r_fmt = score_format(c) if cfg.include_format_reward else 0.0
        out.append(RewardBreakdown(value=r_val, efficiency=r_eff, calibration=r_cal, format=r_fmt))
    return out


def _component_reward(component: str, cfg: BETRewardConfig, prompts, completions, answer, **kwargs):
    breakdowns = compute_bet_rewards(prompts, completions, answer, cfg)
    return [getattr(b, component) for b in breakdowns]


def make_trl_reward_functions(config: BETRewardConfig | None = None) -> List[Any]:
    """Return a list of reward functions compatible with TRL's GRPOTrainer."""
    cfg = config or BETRewardConfig()
    functions = [
        partial(_component_reward, "value", cfg),
        partial(_component_reward, "efficiency", cfg),
        partial(_component_reward, "calibration", cfg),
    ]
    if cfg.include_format_reward:
        functions.insert(0, partial(_component_reward, "format", cfg))
    return functions
