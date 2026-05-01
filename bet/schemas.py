from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ParsedResponse:
    text: str
    format_ok: bool
    solvability_pred: Optional[float]
    budget: Optional[float]
    think: str
    boxed: Optional[str]
    is_fold: bool
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupProfile:
    prompt_key: str
    n: int
    num_correct: int
    solvability: float
    efficient_cost: float
    budget_target: float
    correct_lengths: List[float]

    @property
    def is_zero_return(self) -> bool:
        return self.num_correct == 0


@dataclass
class RewardBreakdown:
    value: float
    efficiency: float
    calibration: float
    format: float = 0.0

    @property
    def total(self) -> float:
        return self.value + self.efficiency + self.calibration + self.format

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d
