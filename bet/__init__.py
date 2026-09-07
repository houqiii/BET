"""Budget-efficient reasoning utilities."""

from .group_constraint import (
    apply_guaranteed_attempt,
    declared_abstentions,
    force_attempt_declaration,
    guaranteed_attempt_index,
)

__version__ = "0.1.0"

__all__ = [
    "apply_guaranteed_attempt",
    "declared_abstentions",
    "force_attempt_declaration",
    "guaranteed_attempt_index",
    "__version__",
]
