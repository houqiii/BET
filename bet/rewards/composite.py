"""Composite BET reward R(y|x) = R_VAL + R_EFF + R_CAL."""
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
    max_completion_tokens: int = 16384
    efficient_cost_percentile: float = 0.30
    delta: float = 0.10
    lambda_abstain: float = 0.80
    alpha_fail: float = 0.20
    beta: float = 0.30
    tau: float = 0.20
    gamma_s: float = 0.10
    gamma_b: float = 0.20
    gamma_s_unsolvable: float = 0.20
    gamma_b_unsolvable: float = 0.10
    mu: float = 2.0
    include_format_reward: bool = True


def compute_bet_rewards(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    answers: Sequence[Any],
    config: BETRewardConfig | None = None,
    overridden: Sequence[bool] | None = None,
) -> List[RewardBreakdown]:
    cfg = config or BETRewardConfig()
    flags = list(overridden) if overridden is not None else [False] * len(list(completions))
    profiles = compute_group_profiles(
        prompts,
        completions,
        answers,
        max_completion_tokens=cfg.max_completion_tokens,
        efficient_cost_percentile=cfg.efficient_cost_percentile,
    )
    out: List[RewardBreakdown] = []
    for i, (p, c, a) in enumerate(zip(prompts, completions, answers)):
        r_val = score_value(
            p, c, a, profiles,
            delta=cfg.delta,
            lambda_abstain=cfg.lambda_abstain,
            alpha_fail=cfg.alpha_fail,
            max_completion_tokens=cfg.max_completion_tokens,
        )
        r_eff = score_efficiency(p, c, a, profiles, beta=cfg.beta, tau=cfg.tau)
        if i < len(flags) and flags[i]:
            r_cal = 0.0
        else:
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


def _component_reward(component: str, cfg: BETRewardConfig, prompts, completions, answer, overridden=None, **kwargs):
    breakdowns = compute_bet_rewards(prompts, completions, answer, cfg, overridden=overridden)
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
