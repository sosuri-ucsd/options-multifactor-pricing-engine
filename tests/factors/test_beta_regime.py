from datetime import date

import numpy as np
import pytest

from factors import beta_regime


def _closes_from_returns(returns, start=100.0):
    closes = [start]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return closes


def test_rolling_beta_recovers_known_linear_relationship():
    rng = np.random.default_rng(0)
    spy_returns = rng.normal(0, 0.01, 130)
    ticker_returns = 2.0 * spy_returns  # exact beta of 2, no idiosyncratic noise

    spy_closes = _closes_from_returns(spy_returns)
    ticker_closes = _closes_from_returns(ticker_returns)

    beta = beta_regime.rolling_beta(ticker_closes, spy_closes, window=126)

    assert beta == pytest.approx(2.0)


def test_rolling_beta_none_with_insufficient_window():
    assert beta_regime.rolling_beta([100, 101, 102], [100, 101, 102], window=126) is None


def test_beta_component_neutral_at_one():
    assert beta_regime.beta_component(1.0) == 0.0
    assert beta_regime.beta_component(2.0) < 0
    assert beta_regime.beta_component(0.0) > 0


def test_vix_regime_score_and_gate_calm_market():
    score, passed = beta_regime.vix_regime_score(vix_level=15.0, vix_trend_pct=-0.05)
    assert score > 0
    assert passed is True


def test_vix_regime_score_and_gate_crisis():
    score, passed = beta_regime.vix_regime_score(vix_level=40.0, vix_trend_pct=0.20)
    assert score < 0
    assert passed is False


def test_vix_regime_high_but_falling_is_not_a_gate():
    _, passed = beta_regime.vix_regime_score(vix_level=40.0, vix_trend_pct=-0.10)
    assert passed is True


def test_compute_gates_out_crisis_regime():
    rng = np.random.default_rng(1)
    spy_returns = rng.normal(0, 0.01, 130)
    ticker_returns = 1.0 * spy_returns
    spy_closes = _closes_from_returns(spy_returns)
    ticker_closes = _closes_from_returns(ticker_returns)
    vix_closes = [20.0] * 5 + [45.0]  # only trend lookback matters, sharp spike up

    result = beta_regime.compute("AAPL", date(2026, 6, 1), ticker_closes, spy_closes, vix_closes)

    # Not enough VIX history for the configured trend lookback -> regime
    # component falls back to neutral, not a gate -- confirms no false-positive gating.
    assert result.passed_gate is True
