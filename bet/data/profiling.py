from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..constants import UNSOLVABLE_TOKEN


@dataclass
class ProfileRecord:
    problem: str
    answer: str
    regime: str
    solvability: float
    efficient_cost: float
    selected_trace: str
    selected_answer: str


def profile_to_sft_target(record: ProfileRecord) -> Dict[str, Any]:
    if record.regime == 'nice_fold' or record.solvability == 0:
        difficulty = 0.96
        budget = 0.08
        think = 'This query is beyond my current reliable capability.'
        answer = UNSOLVABLE_TOKEN
    elif record.regime == 'hero_call':
        difficulty = 0.78
        budget = min(0.95, max(0.45, record.efficient_cost / 16384.0))
        think = record.selected_trace
        answer = record.selected_answer
    else:
        difficulty = 0.18
        budget = min(0.35, max(0.10, record.efficient_cost / 16384.0))
        think = record.selected_trace
        answer = record.selected_answer

    completion = f"""<predict>
Difficulty: {difficulty:.2f}
Budget: {budget:.2f}
</predict>
<think>
{think.strip()}
</think>
\\boxed{{{answer}}}"""
    return {
        'problem': record.problem,
        'completion': completion,
        'metadata': {'regime': record.regime, 's_hat': record.solvability, 'c_star': record.efficient_cost},
    }
