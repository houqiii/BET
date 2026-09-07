"""Two-phase group rollout: declare in <predict>, then complete the trace."""
from __future__ import annotations

import random
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..constants import FORCED_ATTEMPT_BUDGET
from ..group_constraint import apply_guaranteed_attempt

GenerateFn = Callable[[Sequence[str]], List[str]]


def generate_group(
    prompt: str,
    *,
    declare: GenerateFn,
    complete: GenerateFn,
    k: int,
    rng: Optional[random.Random] = None,
    forced_budget: float = FORCED_ATTEMPT_BUDGET,
) -> Tuple[List[str], List[bool]]:
    declarations = list(declare([prompt] * k))
    declarations, overridden = apply_guaranteed_attempt(
        declarations, k=k, rng=rng, forced_budget=forced_budget,
    )
    continuations = list(complete([prompt + d for d in declarations]))
    completions = [d + c for d, c in zip(declarations, continuations)]
    return completions, overridden


def generate_groups(
    prompts: Sequence[str],
    *,
    declare: GenerateFn,
    complete: GenerateFn,
    k: int,
    rng: Optional[random.Random] = None,
    forced_budget: float = FORCED_ATTEMPT_BUDGET,
) -> Tuple[List[str], List[str], List[bool]]:
    flat_prompts: List[str] = []
    flat_completions: List[str] = []
    flat_overridden: List[bool] = []
    for prompt in prompts:
        completions, overridden = generate_group(
            prompt, declare=declare, complete=complete, k=k,
            rng=rng, forced_budget=forced_budget,
        )
        flat_prompts.extend([prompt] * k)
        flat_completions.extend(completions)
        flat_overridden.extend(overridden)
    return flat_prompts, flat_completions, flat_overridden
