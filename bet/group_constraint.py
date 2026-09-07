"""Group-level constraint that keeps one attempt in every rollout group."""
from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence

from .constants import FORCED_ATTEMPT_BUDGET, PREDICT_END, PREDICT_START
from .parsing import declared_abstention, get_text


def declared_abstentions(completions: Sequence[Any]) -> List[bool]:
    return [declared_abstention(c) for c in completions]


def guaranteed_attempt_index(
    abstentions: Sequence[bool],
    rng: Optional[random.Random] = None,
) -> Optional[int]:
    if not abstentions or not all(abstentions):
        return None
    return (rng or random).randrange(len(abstentions))


def force_attempt_declaration(
    declaration: Any,
    *,
    solvability: float,
    budget: float = FORCED_ATTEMPT_BUDGET,
) -> str:
    text = get_text(declaration)
    block = (
        f"{PREDICT_START}\n"
        f"Solvability: {min(1.0, max(0.0, solvability)):.2f}\n"
        f"Budget: {min(1.0, max(0.0, budget)):.2f}\n"
        f"{PREDICT_END}"
    )
    end = text.find(PREDICT_END)
    if end < 0:
        return block
    tail = text[end + len(PREDICT_END):]
    start = text.find(PREDICT_START)
    head = text[:start] if start >= 0 else ""
    return head + block + tail


def apply_guaranteed_attempt(
    declarations: Sequence[Any],
    *,
    k: Optional[int] = None,
    rng: Optional[random.Random] = None,
    forced_budget: float = FORCED_ATTEMPT_BUDGET,
) -> tuple[List[str], List[bool]]:
    texts = [get_text(d) for d in declarations]
    overridden = [False] * len(texts)
    index = guaranteed_attempt_index(declared_abstentions(texts), rng)
    if index is not None:
        size = k or len(texts)
        texts[index] = force_attempt_declaration(
            texts[index],
            solvability=1.0 / max(1, size),
            budget=forced_budget,
        )
        overridden[index] = True
    return texts, overridden
