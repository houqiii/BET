from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..constants import UNSOLVABLE_TOKEN
from ..parsing import clamp01


@dataclass
class ProfileRecord:
    problem: str
    answer: str
    regime: str
    solvability: float
    efficient_cost: float
    selected_trace: str
    selected_answer: str
    difficulty: Optional[float] = None


def behavioral_solvability(
    difficulty: float,
    cost_fraction: float,
    *,
    difficulty_weight: float = 0.6,
    cost_weight: float = 0.4,
) -> float:
    d = clamp01(difficulty)
    c = clamp01(cost_fraction)
    return round(clamp01(1.0 - difficulty_weight * d - cost_weight * c), 2)


def profile_to_sft_target(
    record: ProfileRecord,
    max_completion_tokens: float = 16384.0,
) -> Dict[str, Any]:
    """Render one cold-start demonstration in the BET output template."""
    if record.regime == 'nice_fold' or record.solvability == 0:
        s_pred = 0.0
        b_pred = 0.0
        think = 'This query is beyond my current reliable capability.'
        answer = UNSOLVABLE_TOKEN
    else:
        b_pred = clamp01(record.efficient_cost / max_completion_tokens)
        if record.difficulty is None:
            s_pred = record.solvability
        else:
            s_pred = behavioral_solvability(record.difficulty, b_pred)
        think = record.selected_trace
        answer = record.selected_answer

    completion = f"""<predict>
Solvability: {s_pred:.2f}
Budget: {b_pred:.2f}
</predict>
<think>
{think.strip()}
</think>
\\boxed{{{answer}}}"""
    return {
        'problem': record.problem,
        'completion': completion,
        'metadata': {
            'regime': record.regime,
            's_hat': s_pred,
            'c_star': record.efficient_cost,
            'difficulty': record.difficulty,
        },
    }
