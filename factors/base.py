"""
Shared interface every factor module implements.

Each factor is independently testable and independently backtestable: given
a ticker and an as_of date, it returns a FactorResult carrying a normalized
score plus every raw input that fed into it, so any score in a later ranked
output can be traced back to the numbers that produced it.

Score convention: score is a float clipped to [-1, 1].
    +1  = most favorable for the initial strategies (selling premium via
          covered calls / cash-secured puts) -- e.g. richly priced vol,
          favorable regime, strong liquidity.
    -1  = least favorable / avoid.
     0  = neutral / no edge either way.
This lets the decision layer combine factor scores with simple weighted
averaging without each factor needing to know how the others are scaled.

`passed_gate` is only meaningful for hard-gate factors (liquidity always;
beta_regime in a crisis regime): False means the contract/underlying must
be excluded from the ranked output regardless of its score. Soft-signal
factors always leave it True.
"""
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Any


def clip_score(value: float) -> float:
    return max(-1.0, min(1.0, value))


@dataclass
class FactorResult:
    factor_name: str
    ticker: str
    as_of: date_type
    score: float
    raw_inputs: dict[str, Any] = field(default_factory=dict)
    passed_gate: bool = True

    def __post_init__(self) -> None:
        self.score = clip_score(self.score)
