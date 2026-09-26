from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..constants import UNSOLVABLE_TOKEN
from ..math_eval import canonical_gold
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
        b_pred = clamp01(max(1.0, record.efficient_cost) / max_completion_tokens)
        if record.difficulty is None:
            s_pred = behavioral_solvability(0.0, b_pred)
        else:
            s_pred = behavioral_solvability(record.difficulty, b_pred)
        think = record.selected_trace
        answer = canonical_gold(record.selected_answer)

    completion = f"""<predict>
Solvability: {s_pred:.2f}
Budget: {b_pred:.6f}
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


def build_cold_start_profiles(samples, *, tokenizer=None, split_fraction=0.6):
    from ..math_eval import is_correct
    from ..parsing import parse_response, think_token_proxy

    if not 0 <= split_fraction <= 1:
        raise ValueError('split_fraction must be in [0, 1].')
    records = []
    for sample in samples:
        completion = sample['completion']
        correct = is_correct(completion, sample['answer'])
        parsed = parse_response(completion)
        trace = parsed.think
        if not trace and '</think>' in completion:
            trace = completion.split('</think>', 1)[0].split('<think>', 1)[-1].strip()
        if correct and not trace:
            raise ValueError('Correct cold-start samples must include their reasoning trace.')
        cost = think_token_proxy(completion, tokenizer)
        difficulty = sample.get('difficulty')
        if difficulty is not None and not 0 <= float(difficulty) <= 1:
            raise ValueError('Normalize difficulty to [0, 1] before constructing behavioral targets.')
        records.append(ProfileRecord(
            problem=sample['problem'], answer=sample['answer'],
            regime='short_solve' if correct else 'nice_fold',
            solvability=1.0 if correct else 0.0,
            efficient_cost=cost if correct else 0.0,
            selected_trace=trace,
            selected_answer=canonical_gold(sample['answer']),
            difficulty=None if difficulty is None else float(difficulty),
        ))
    costs = sorted(r.efficient_cost for r in records if r.solvability > 0)
    if costs:
        position = (len(costs) - 1) * split_fraction
        lower = int(position)
        upper = min(lower + 1, len(costs) - 1)
        threshold = costs[lower] + (position - lower) * (costs[upper] - costs[lower])
        for record in records:
            if record.solvability > 0 and record.efficient_cost > threshold:
                record.regime = 'hero_call'
    return records
