"""
Momentum/technical factor (lower priority per spec) -- a basic trend signal
used as a directional sanity check before writing calls into a breakout,
not a standalone alpha source.

Measures how large the underlying's trailing move is relative to its own
typical daily variability, as a z-score:

    return_N       = close_t / close_{t-N} - 1
    daily_vol      = stdev of trailing daily log returns
    expected_vol_N = daily_vol * sqrt(N)          (random-walk scaling)
    z              = return_N / expected_vol_N

|z| large means the underlying has moved much further over the lookback
than its own recent volatility would suggest is "normal" -- i.e. it's in
the middle of a trending breakout, in either direction. That is scored as
*unfavorable* for the initial single-leg strategies here: a strong move can
keep running through a strike that looked comfortably OTM a few days ago
(a covered call getting run over to the upside, or a cash-secured put
strike getting broken to the downside). |z| small (range-bound) is scored
favorable, since strikes are less likely to be blown through by pure trend
continuation:

    score = clip(1 - |z| / MOMENTUM_ZSCORE_SCALE)
"""
import math
import statistics
from datetime import date as date_type
from typing import Optional

import config
from factors.base import FactorResult, clip_score


def _daily_log_returns(closes: list[float]) -> list[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]


def momentum_zscore(
    closes: list[float], lookback: int = config.MOMENTUM_LOOKBACK_DAYS
) -> Optional[float]:
    """closes: chronological daily closes ending on as_of, needs at least
    lookback + 1 points for the return and enough of those for a vol estimate."""
    if len(closes) < lookback + 1:
        return None
    window_return = closes[-1] / closes[-(lookback + 1)] - 1.0
    daily_returns = _daily_log_returns(closes[-(lookback + 1) :])
    if len(daily_returns) < 2:
        return None
    daily_vol = statistics.pstdev(daily_returns)
    if daily_vol == 0:
        return None
    expected_vol_n = daily_vol * math.sqrt(lookback)
    return window_return / expected_vol_n


def compute(ticker: str, as_of: date_type, closes: list[float]) -> FactorResult:
    z = momentum_zscore(closes)
    if z is None:
        return FactorResult(
            factor_name="momentum",
            ticker=ticker,
            as_of=as_of,
            score=0.0,
            raw_inputs={"reason": "insufficient price history"},
        )

    score = clip_score(1.0 - abs(z) / config.MOMENTUM_ZSCORE_SCALE)

    return FactorResult(
        factor_name="momentum",
        ticker=ticker,
        as_of=as_of,
        score=score,
        raw_inputs={"zscore": z, "lookback_days": config.MOMENTUM_LOOKBACK_DAYS},
    )
