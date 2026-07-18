from datetime import date

from decision.edge import Candidate
from decision.ranking import passes_all_gates, rank_candidates, to_audit_rows
from factors.base import FactorResult


def _factor(name, score, passed_gate=True):
    return FactorResult(
        factor_name=name, ticker="AAPL", as_of=date(2026, 6, 1), score=score, passed_gate=passed_gate
    )


def _candidate(ticker, market_price, model_price, capital_required, factor_results):
    return Candidate(
        ticker=ticker,
        strategy="cash_secured_put",
        option_type="put",
        strike=140.0,
        expiration=date(2026, 7, 17),
        dte=20,
        market_price=market_price,
        model_price=model_price,
        capital_required=capital_required,
        factor_results=factor_results,
    )


def test_passes_all_gates_true_when_no_gate_fails():
    c = _candidate("AAPL", 3.0, 2.0, 14_000, [_factor("liquidity", 0.5), _factor("beta_regime", -0.2)])
    assert passes_all_gates(c)


def test_passes_all_gates_false_when_any_gate_fails():
    c = _candidate(
        "AAPL", 3.0, 2.0, 14_000, [_factor("liquidity", 0.5, passed_gate=False), _factor("beta_regime", 0.1)]
    )
    assert not passes_all_gates(c)


def test_rank_candidates_excludes_gated_candidates():
    good = _candidate("AAPL", 3.0, 2.0, 14_000, [_factor("liquidity", 0.5)])
    gated = _candidate("MEME", 5.0, 2.0, 10_000, [_factor("liquidity", -0.9, passed_gate=False)])

    ranked = rank_candidates([good, gated])

    tickers = [r.candidate.ticker for r in ranked]
    assert "MEME" not in tickers
    assert "AAPL" in tickers


def test_rank_candidates_sorts_descending_by_expected_pnl_per_capital():
    low_edge = _candidate("LOW", 2.10, 2.00, 14_000, [_factor("liquidity", 0.1)])
    high_edge = _candidate("HIGH", 3.00, 2.00, 14_000, [_factor("liquidity", 0.1)])

    ranked = rank_candidates([low_edge, high_edge])

    assert [r.candidate.ticker for r in ranked] == ["HIGH", "LOW"]
    assert ranked[0].expected_pnl_per_capital > ranked[1].expected_pnl_per_capital


def test_to_audit_rows_includes_contributing_factors():
    c = _candidate(
        "AAPL", 3.0, 2.0, 14_000, [_factor("liquidity", 0.5), _factor("skew", 0.3)]
    )
    ranked = rank_candidates([c])

    rows = to_audit_rows(ranked)

    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAPL"
    assert row["strategy"] == "cash_secured_put"
    assert row["contributing_factors"] == {"liquidity": 0.5, "skew": 0.3}
    assert "expected_pnl_per_capital" in row
    assert "model_minus_market_edge" in row
