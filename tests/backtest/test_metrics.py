from datetime import date, timedelta

import pytest

from backtest.metrics import (
    Trade,
    average_holding_period_days,
    max_drawdown,
    model_calibration,
    pnl_attribution_by_factor,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    win_rate,
)


def _trade(pnl, expected, holding_days=5, factors=None, ticker="AAPL"):
    entry = date(2026, 1, 1)
    return Trade(
        ticker=ticker,
        entry_date=entry,
        exit_date=entry + timedelta(days=holding_days),
        realized_pnl=pnl,
        expected_edge_pnl=expected,
        capital_required=10_000.0,
        contributing_factors=factors or {},
    )


def test_sharpe_ratio_zero_for_constant_returns_with_zero_stdev():
    assert sharpe_ratio([0.01, 0.01, 0.01]) == 0.0


def test_sharpe_ratio_positive_for_positive_mean_returns():
    returns = [0.01, -0.005, 0.02, 0.0, 0.015]
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_insufficient_data_returns_zero():
    assert sharpe_ratio([0.01]) == 0.0


def test_sortino_ignores_upside_volatility():
    # Same mean, but one series has upside swings, one has downside swings.
    upside_swings = [0.10, -0.01, 0.10, -0.01]
    downside_swings = [-0.10, 0.01, -0.10, 0.01]
    assert sortino_ratio(upside_swings) > sortino_ratio(downside_swings)


def test_max_drawdown_flat_curve_is_zero():
    assert max_drawdown([1.0, 1.0, 1.0]) == 0.0


def test_max_drawdown_detects_peak_to_trough():
    curve = [1.0, 1.2, 0.9, 1.1]
    dd = max_drawdown(curve)
    assert dd == pytest.approx((0.9 - 1.2) / 1.2)


def test_win_rate():
    trades = [_trade(100, 50), _trade(-20, 50), _trade(30, 50)]
    assert win_rate(trades) == pytest.approx(2 / 3)


def test_win_rate_empty_is_zero():
    assert win_rate([]) == 0.0


def test_average_holding_period_days():
    trades = [_trade(1, 1, holding_days=5), _trade(1, 1, holding_days=15)]
    assert average_holding_period_days(trades) == 10.0


def test_pnl_attribution_splits_by_factor_score_share():
    trades = [_trade(100, 50, factors={"vol_richness": 0.6, "skew": 0.4})]
    attribution = pnl_attribution_by_factor(trades)
    assert attribution["vol_richness"] == pytest.approx(60.0)
    assert attribution["skew"] == pytest.approx(40.0)


def test_pnl_attribution_skips_trades_with_zero_total_score():
    trades = [_trade(100, 50, factors={"vol_richness": 0.0})]
    assert pnl_attribution_by_factor(trades) == {}


def test_model_calibration_reports_realized_minus_expected():
    trades = [_trade(100, 50), _trade(-20, 30)]
    calibration = model_calibration(trades)
    assert calibration["total_expected_pnl"] == 80
    assert calibration["total_realized_pnl"] == 80
    assert calibration["realized_minus_expected"] == 0


def test_summarize_bundles_everything():
    trades = [_trade(100, 50, factors={"skew": 1.0})]
    summary = summarize(trades, periodic_returns=[0.01, -0.005, 0.02])
    assert summary["n_trades"] == 1
    assert "sharpe_ratio" in summary
    assert "sortino_ratio" in summary
    assert "max_drawdown" in summary
    assert "pnl_attribution_by_factor" in summary


def test_summarize_without_periodic_returns_omits_sharpe():
    trades = [_trade(100, 50)]
    summary = summarize(trades)
    assert "sharpe_ratio" not in summary
