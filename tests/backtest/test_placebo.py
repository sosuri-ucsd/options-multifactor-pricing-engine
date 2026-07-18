from datetime import date, timedelta

import numpy as np

from backtest.placebo import (
    CandidateOutcome,
    placebo_comparison,
    run_selection_backtest,
    select_random_n,
    select_top_n_by_expected_edge,
)


def _candidate(ticker, expected_edge, realized_pnl):
    entry = date(2026, 1, 1)
    return CandidateOutcome(
        ticker=ticker,
        entry_date=entry,
        exit_date=entry + timedelta(days=10),
        expected_edge_pnl=expected_edge,
        realized_pnl=realized_pnl,
        capital_required=10_000.0,
        contributing_factors={"skew": 0.5},
    )


def test_select_top_n_by_expected_edge_picks_highest():
    candidates = [_candidate("A", 10, 1), _candidate("B", 50, 2), _candidate("C", 30, 3)]
    top1 = select_top_n_by_expected_edge(candidates, 1)
    assert [c.ticker for c in top1] == ["B"]


def test_select_random_n_respects_n_and_no_replacement():
    candidates = [_candidate(str(i), i, i) for i in range(10)]
    rng = np.random.default_rng(0)
    selected = select_random_n(candidates, 4, rng)
    assert len(selected) == 4
    assert len(set(c.ticker for c in selected)) == 4


def test_select_random_n_caps_at_available_candidates():
    candidates = [_candidate("A", 1, 1)]
    rng = np.random.default_rng(0)
    assert len(select_random_n(candidates, 5, rng)) == 1


def test_run_selection_backtest_aggregates_across_periods():
    candidates_by_period = {
        "period1": [_candidate("A", 10, 5), _candidate("B", 20, 8)],
        "period2": [_candidate("C", 30, 12)],
    }
    trades = run_selection_backtest(candidates_by_period, n_per_period=1, selection_fn=select_top_n_by_expected_edge)
    assert len(trades) == 2
    assert {t.ticker for t in trades} == {"B", "C"}


def test_placebo_comparison_returns_real_and_placebo_summaries():
    candidates_by_period = {
        f"period{i}": [
            _candidate(f"good{i}", expected_edge=100, realized_pnl=50),
            _candidate(f"bad{i}", expected_edge=-100, realized_pnl=-50),
        ]
        for i in range(20)
    }

    result = placebo_comparison(candidates_by_period, n_per_period=1, seed=1)

    assert "real" in result and "placebo" in result
    # The real strategy always picks the good candidate -> all wins.
    assert result["real"]["win_rate"] == 1.0
    assert result["real"]["model_calibration"]["total_realized_pnl"] == 50 * 20
