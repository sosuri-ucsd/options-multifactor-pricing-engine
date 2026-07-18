"""
Placebo test: reruns the identical backtest structure with randomized/
shuffled entry signals and reports both results side by side, so it's
visible whether the real signal clears the noise floor rather than being
assumed to.

Concretely: for each period, there is a universe of CandidateOutcome
objects -- every contract the factor/decision pipeline could have entered
that period, each carrying what actually happened had it been entered
(its realized P&L). The *real* backtest selects the top N per period by
the model's expected_edge_pnl (decision/ranking.py's ordering). The
*placebo* backtest selects a random N per period from the same universe,
with no reference to the model's ranking at all. Comparing metrics (from
backtest/metrics.py) between the two answers the actual question -- does
picking by model edge do better than picking blind -- rather than just
reporting the real backtest's Sharpe in isolation and hoping it's good.
"""
from dataclasses import dataclass
from datetime import date as date_type
from typing import Callable

import numpy as np

from backtest.metrics import Trade, summarize


@dataclass
class CandidateOutcome:
    ticker: str
    entry_date: date_type
    exit_date: date_type
    expected_edge_pnl: float
    realized_pnl: float
    capital_required: float
    contributing_factors: dict[str, float]


def _to_trade(candidate: CandidateOutcome) -> Trade:
    return Trade(
        ticker=candidate.ticker,
        entry_date=candidate.entry_date,
        exit_date=candidate.exit_date,
        realized_pnl=candidate.realized_pnl,
        expected_edge_pnl=candidate.expected_edge_pnl,
        capital_required=candidate.capital_required,
        contributing_factors=candidate.contributing_factors,
    )


def select_top_n_by_expected_edge(
    candidates: list[CandidateOutcome], n: int
) -> list[CandidateOutcome]:
    return sorted(candidates, key=lambda c: c.expected_edge_pnl, reverse=True)[:n]


def select_random_n(
    candidates: list[CandidateOutcome], n: int, rng: np.random.Generator
) -> list[CandidateOutcome]:
    if not candidates:
        return []
    n = min(n, len(candidates))
    indices = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in indices]


def run_selection_backtest(
    candidates_by_period: dict,
    n_per_period: int,
    selection_fn: Callable[[list[CandidateOutcome], int], list[CandidateOutcome]],
) -> list[Trade]:
    selected = []
    for candidates in candidates_by_period.values():
        selected.extend(selection_fn(candidates, n_per_period))
    return [_to_trade(c) for c in selected]


def placebo_comparison(
    candidates_by_period: dict,
    n_per_period: int,
    seed: int = 0,
) -> dict:
    """Returns {"real": {...summary}, "placebo": {...summary}} using
    backtest.metrics.summarize on each trade set. periodic_returns isn't
    computed here (this operates on a candidate universe, not a full daily
    equity series) -- callers wanting Sharpe/Sortino should build a returns
    series separately and call backtest.metrics.summarize directly."""
    real_trades = run_selection_backtest(
        candidates_by_period, n_per_period, select_top_n_by_expected_edge
    )

    rng = np.random.default_rng(seed)
    placebo_trades = run_selection_backtest(
        candidates_by_period, n_per_period, lambda c, n: select_random_n(c, n, rng)
    )

    return {
        "real": summarize(real_trades),
        "placebo": summarize(placebo_trades),
    }
