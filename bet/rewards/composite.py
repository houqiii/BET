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
    beta: float = 0.30
    tau: float = 0.25
    gamma_d: float = 0.10
    gamma_b_under: float = 0.20
    gamma_b_over: float = 0.10
    gamma_d_unsolvable: float = 0.40
    gamma_b_unsolvable: float = 0.40
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
            p,
            c,
            a,
            profiles,
            delta=cfg.delta,
            lambda_abstain=cfg.lambda_abstain,
            max_completion_tokens=cfg.max_completion_tokens,
        )
        r_eff = score_efficiency(p, c, a, profiles, beta=cfg.beta, tau=cfg.tau)
        r_cal, _ = score_calibration(
            p,
            c,
            profiles,
            gamma_d=cfg.gamma_d,
            gamma_b_under=cfg.gamma_b_under,
            gamma_b_over=cfg.gamma_b_over,
            gamma_d_unsolvable=cfg.gamma_d_unsolvable,
            gamma_b_unsolvable=cfg.gamma_b_unsolvable,
        )
        r_fmt = score_format(c) if cfg.include_format_reward else 0.0
        out.append(RewardBreakdown(value=r_val, efficiency=r_eff, calibration=r_cal, format=r_fmt))
    return out


def _component_reward(component: str, cfg: BETRewardConfig, prompts, completions, answer, **kwargs):
    breakdowns = compute_bet_rewards(prompts, completions, answer, cfg)
    return [getattr(b, component) for b in breakdowns]


def make_trl_reward_functions(config: BETRewardConfig | None = None) -> List[Any]:
    cfg = config or BETRewardConfig()
    functions = [
        partial(_component_reward, "value", cfg),
        partial(_component_reward, "efficiency", cfg),
        partial(_component_reward, "calibration", cfg),
    ]
    if cfg.include_format_reward:
        functions.insert(0, partial(_component_reward, "format", cfg))
    return functions
