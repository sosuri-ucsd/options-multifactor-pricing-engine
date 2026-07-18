"""
Beta/regime factor: gates which underlyings are even considered and whether
premium-selling is currently favored, via two independent reads.

1. Rolling beta to SPY.
   beta = Cov(ticker_returns, SPY_returns) / Var(SPY_returns)
   over the trailing BETA_ROLLING_WINDOW_DAYS daily log returns. This is
   plain OLS single-factor beta. A short-premium book already takes on
   vol/skew/liquidity risk from the other factors; this component
   additionally penalizes names whose beta sits far from 1 (in either
   direction -- high beta amplifies systematic moves against a short-vol
   position, and this factor scores that as added, uncompensated risk
   rather than as a directional view):

       beta_component = clip(-(beta - 1.0) / BETA_COMPONENT_SCALE)

2. VIX level + trend regime.
   Two reads on the "weather", not the specific ticker:
     - level_component: how far current VIX sits above/below a calm
       baseline (VIX_LEVEL_BASELINE). Elevated VIX = elevated systematic
       tail risk = unfavorable for initiating new short premium.
     - trend_component: % change in VIX over VIX_TREND_LOOKBACK_DAYS.
       Rising VIX (stress building) is scored unfavorable; falling VIX
       (fear subsiding, a regime premium-sellers traditionally favor) is
       scored favorable.
   regime_score = 0.5 * level_component + 0.5 * trend_component

Combined score = 0.5 * beta_component + 0.5 * regime_score.

Hard gate: VIX above VIX_CRISIS_THRESHOLD *and* still rising is a crisis
regime -- passed_gate=False, unconditionally excluding new premium-selling
entries regardless of score, independent of factors/liquidity.py's gate.
"""
import statistics
from datetime import date as date_type
from typing import Optional

import config
from factors.base import FactorResult, clip_score


def _log_returns(closes: list[float]) -> list[float]:
    return [
        (closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else 0.0
        for i in range(1, len(closes))
    ]


def rolling_beta(
    ticker_closes: list[float],
    spy_closes: list[float],
    window: int = config.BETA_ROLLING_WINDOW_DAYS,
) -> Optional[float]:
    """Trailing-window closes (same dates, same length, chronological) for the
    ticker and SPY. Returns None if there isn't a full window of returns."""
    ticker_returns = _log_returns(ticker_closes[-(window + 1) :])
    spy_returns = _log_returns(spy_closes[-(window + 1) :])
    if len(ticker_returns) < window or len(spy_returns) < window:
        return None
    spy_var = statistics.pvariance(spy_returns)
    if spy_var == 0:
        return None
    mean_t, mean_s = statistics.fmean(ticker_returns), statistics.fmean(spy_returns)
    covariance = sum(
        (t - mean_t) * (s - mean_s) for t, s in zip(ticker_returns, spy_returns)
    ) / len(ticker_returns)
    return covariance / spy_var


def beta_component(beta: float) -> float:
    return clip_score(-(beta - 1.0) / config.BETA_COMPONENT_SCALE)


def vix_regime_score(vix_level: float, vix_trend_pct: float) -> tuple[float, bool]:
    """Returns (regime_score, passed_gate)."""
    level_component = clip_score(-(vix_level - config.VIX_LEVEL_BASELINE) / config.VIX_LEVEL_SCALE)
    trend_component = clip_score(-vix_trend_pct / config.VIX_TREND_SCALE)
    regime_score = 0.5 * level_component + 0.5 * trend_component

    is_crisis = vix_level > config.VIX_CRISIS_THRESHOLD and vix_trend_pct > 0
    return regime_score, not is_crisis


def compute(
    ticker: str,
    as_of: date_type,
    ticker_closes: list[float],
    spy_closes: list[float],
    vix_closes: list[float],
) -> FactorResult:
    """*_closes: chronological daily closes ending on as_of. vix_closes needs at
    least VIX_TREND_LOOKBACK_DAYS + 1 points to compute the trend."""
    beta = rolling_beta(ticker_closes, spy_closes)
    beta_comp = beta_component(beta) if beta is not None else 0.0

    raw_inputs: dict = {"beta": beta, "beta_component": beta_comp}

    passed_gate = True
    if len(vix_closes) >= config.VIX_TREND_LOOKBACK_DAYS + 1:
        vix_level = vix_closes[-1]
        vix_prior = vix_closes[-(config.VIX_TREND_LOOKBACK_DAYS + 1)]
        vix_trend_pct = (vix_level - vix_prior) / vix_prior if vix_prior else 0.0
        regime_score, passed_gate = vix_regime_score(vix_level, vix_trend_pct)
        raw_inputs.update(
            {
                "vix_level": vix_level,
                "vix_trend_pct": vix_trend_pct,
                "regime_score": regime_score,
            }
        )
    else:
        regime_score = 0.0
        raw_inputs["reason_regime"] = "insufficient VIX history for trend"

    score = 0.5 * beta_comp + 0.5 * regime_score

    return FactorResult(
        factor_name="beta_regime",
        ticker=ticker,
        as_of=as_of,
        score=score,
        raw_inputs=raw_inputs,
        passed_gate=passed_gate,
    )
