"""Cold-start data construction (Algorithm 1)."""
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


def profile_to_sft_target(
    record: ProfileRecord,
    max_completion_tokens: float = 16384.0,
) -> Dict[str, Any]:
    """Convert a profiled record into an SFT demonstration (Algorithm 1).

    The <predict> block carries s_hat(x) and b*(x) directly,
    matching the paper's Solvability/Budget format.
    """
    if record.regime == 'nice_fold' or record.solvability == 0:
        # Zero-return regime: s_hat ≈ 0, b* = 0
        s_pred = 0.0
        b_pred = 0.0
        think = 'This query is beyond my current reliable capability.'
        answer = UNSOLVABLE_TOKEN
    else:
        s_pred = record.solvability
        b_pred = min(1.0, max(0.0, record.efficient_cost / max_completion_tokens))
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
            's_hat': record.solvability,
            'c_star': record.efficient_cost,
        },
    }
