"""
Backtest performance metrics: Sharpe, Sortino, max drawdown, win rate,
average holding period, and P&L attribution by factor.

Sharpe / Sortino derivation: given a series of per-period returns r_t,

    Sharpe  = mean(r_t - rf_period) / stdev(r_t)              * sqrt(periods_per_year)
    Sortino = mean(r_t - rf_period) / downside_stdev(r_t)     * sqrt(periods_per_year)

where downside_stdev only includes the negative part of (r_t - rf_period)
-- Sortino doesn't penalize upside volatility the way Sharpe's plain stdev
does, which matters here since a premium-selling strategy's return
distribution is expected to be negatively skewed (frequent small wins,
occasional large losses) and Sharpe alone can make that shape look better
than it is.

Max drawdown: largest peak-to-trough decline in a cumulative equity curve,
expressed as a negative fraction of the peak.

P&L attribution by factor: NOT a causal decomposition (that would need
something like a Shapley-value analysis over the ranking function) -- it's
a simpler, fully auditable heuristic: each trade's realized P&L is split
across the factors that contributed to it, in proportion to each factor's
share of the total |score| across all factors that fired on that trade.
Summed across all trades, this gives a rough "how much of the realized P&L
lines up with each factor being right" view, useful for spotting a factor
that's consistently on the wrong side even while the blended signal
nets positive.
"""
import statistics
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Optional

import config


@dataclass
class Trade:
    ticker: str
    entry_date: date_type
    exit_date: date_type
    realized_pnl: float
    expected_edge_pnl: float
    capital_required: float
    contributing_factors: dict[str, float] = field(default_factory=dict)

    @property
    def holding_period_days(self) -> int:
        return (self.exit_date - self.entry_date).days


def sharpe_ratio(
    returns: list[float],
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = config.BACKTEST_TRADING_DAYS_PER_YEAR,
) -> float:
    if len(returns) < 2:
        return 0.0
    rf_period = risk_free_rate_annual / periods_per_year
    excess = [r - rf_period for r in returns]
    stdev = statistics.pstdev(excess)
    if stdev == 0:
        return 0.0
    return statistics.fmean(excess) / stdev * (periods_per_year**0.5)


def sortino_ratio(
    returns: list[float],
    risk_free_rate_annual: float = 0.0,
    periods_per_year: int = config.BACKTEST_TRADING_DAYS_PER_YEAR,
) -> float:
    if len(returns) < 2:
        return 0.0
    rf_period = risk_free_rate_annual / periods_per_year
    excess = [r - rf_period for r in returns]
    downside_sq = [min(0.0, e) ** 2 for e in excess]
    downside_stdev = statistics.fmean(downside_sq) ** 0.5
    if downside_stdev == 0:
        return 0.0
    return statistics.fmean(excess) / downside_stdev * (periods_per_year**0.5)


def max_drawdown(equity_curve: list[float]) -> float:
    """Returns a negative fraction (e.g. -0.23 for a 23% drawdown), or 0.0
    for a curve that never dips below its running peak."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            drawdown = (value - peak) / peak
            worst = min(worst, drawdown)
    return worst


def win_rate(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.realized_pnl > 0)
    return wins / len(trades)


def average_holding_period_days(trades: list[Trade]) -> float:
    if not trades:
        return 0.0
    return statistics.fmean(t.holding_period_days for t in trades)


def pnl_attribution_by_factor(trades: list[Trade]) -> dict[str, float]:
    attribution: dict[str, float] = {}
    for trade in trades:
        total_abs_score = sum(abs(s) for s in trade.contributing_factors.values())
        if total_abs_score == 0:
            continue
        for factor_name, score in trade.contributing_factors.items():
            weight = abs(score) / total_abs_score
            attribution[factor_name] = attribution.get(factor_name, 0.0) + weight * trade.realized_pnl
    return attribution


def model_calibration(trades: list[Trade]) -> dict:
    """Ex-ante expected P&L (what the model's edge estimate said the trade
    was worth) vs. what actually realized, in aggregate -- the basic check
    for whether the pricing model's edge estimates are calibrated or
    systematically over/under-confident."""
    total_expected = sum(t.expected_edge_pnl for t in trades)
    total_realized = sum(t.realized_pnl for t in trades)
    return {
        "total_expected_pnl": total_expected,
        "total_realized_pnl": total_realized,
        "realized_minus_expected": total_realized - total_expected,
    }


def summarize(trades: list[Trade], periodic_returns: Optional[list[float]] = None) -> dict:
    """One-call summary bundling every metric above. periodic_returns (e.g.
    daily portfolio returns) is optional -- Sharpe/Sortino need a return
    series, not just a trade list, since trade-level P&L alone doesn't
    capture holding-period overlap or account sizing."""
    summary = {
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "average_holding_period_days": average_holding_period_days(trades),
        "pnl_attribution_by_factor": pnl_attribution_by_factor(trades),
        "model_calibration": model_calibration(trades),
    }
    if periodic_returns:
        summary["sharpe_ratio"] = sharpe_ratio(periodic_returns)
        summary["sortino_ratio"] = sortino_ratio(periodic_returns)
        equity_curve = [1.0]
        for r in periodic_returns:
            equity_curve.append(equity_curve[-1] * (1 + r))
        summary["max_drawdown"] = max_drawdown(equity_curve)
    return summary
