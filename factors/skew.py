"""
Skew factor: 25-delta risk reversal relative to the ticker's own historical
skew norm.

The 25-delta risk reversal (RR25) is a standard vol-surface summary:

    RR25 = IV(25-delta call) - IV(25-delta put)

For equities RR25 is typically negative -- out-of-the-money puts trade at a
higher implied vol than equidistant calls, reflecting a persistent demand
for crash protection (the "volatility skew" / "smirk"). The raw level of
RR25 mixes together (a) the generic equity put-skew premium that's present
almost all the time and (b) whatever is unusual about this ticker right
now, so this factor scores the second part: how far today's RR25 sits from
this ticker's own historical RR25, as a z-score.

    z = (RR25_today - mean(RR25_history)) / std(RR25_history)

As with vol_richness, no vendor hands back a ready-made historical skew
series, so this module accumulates its own (record_daily_skew /
load_skew_history), same pattern as factors/vol_richness.py.

Sign convention: skew steepening beyond its historical norm (RR25 more
negative than usual, i.e. z very negative) means put IV is unusually rich
relative to calls for this name right now -- favorable for cash-secured-put
premium capture, so that maps to a *positive* score. Skew flattening or
inverting relative to norm (z positive) is scored negative.
"""
import statistics
from datetime import date as date_type
from typing import Optional

import config
from data import cache
from factors.base import FactorResult, clip_score

_HISTORY_KEY_PREFIX = "skew_history"
# A skew move of 1.5 standard deviations from its own historical norm maps
# to a full +-1 score.
_ZSCORE_SCALE = 1.5


def _history_key(ticker: str) -> str:
    return f"{_HISTORY_KEY_PREFIX}:{ticker.upper()}"


def record_daily_skew(ticker: str, as_of: date_type, rr25: float) -> None:
    key = _history_key(ticker)
    history = cache.get(key) or []
    history = [h for h in history if h["date"] != as_of.isoformat()]
    history.append({"date": as_of.isoformat(), "rr25": rr25})
    history.sort(key=lambda h: h["date"])
    cache.set(key, history)


def load_skew_history(
    ticker: str, as_of: date_type, lookback_days: int = config.SKEW_HISTORY_LOOKBACK_DAYS
) -> list[float]:
    history = cache.get(_history_key(ticker)) or []
    cutoff = as_of.toordinal() - lookback_days
    return [
        h["rr25"]
        for h in history
        if cutoff <= date_type.fromisoformat(h["date"]).toordinal() <= as_of.toordinal()
    ]


def _closest_by_delta(
    contracts: list[dict], option_type: str, target_delta: float
) -> Optional[dict]:
    """contracts: Tradier-style chain entries with "option_type" and "greeks.delta".
    Puts carry negative delta in Tradier's convention, so we match on the
    signed target for puts and the unsigned target for calls."""
    signed_target = -target_delta if option_type == "put" else target_delta
    candidates = [
        c
        for c in contracts
        if c.get("option_type") == option_type and c.get("greeks", {}).get("delta") is not None
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c["greeks"]["delta"] - signed_target))


def risk_reversal_25d(
    chain: list[dict], target_delta: float = config.SKEW_DELTA_TARGET
) -> Optional[float]:
    """chain: one expiration's Tradier-style chain (calls and puts mixed,
    each with greeks.delta and greeks.mid_iv/smv_vol)."""
    call = _closest_by_delta(chain, "call", target_delta)
    put = _closest_by_delta(chain, "put", target_delta)
    if call is None or put is None:
        return None
    call_iv = call["greeks"].get("mid_iv") or call["greeks"].get("smv_vol")
    put_iv = put["greeks"].get("mid_iv") or put["greeks"].get("smv_vol")
    if call_iv is None or put_iv is None:
        return None
    return float(call_iv) - float(put_iv)


def compute(ticker: str, as_of: date_type, chain: list[dict]) -> FactorResult:
    rr25 = risk_reversal_25d(chain)
    raw_inputs: dict = {"rr25_today": rr25}

    if rr25 is None:
        raw_inputs["reason"] = "could not identify 25-delta call/put in chain"
        return FactorResult(
            factor_name="skew", ticker=ticker, as_of=as_of, score=0.0, raw_inputs=raw_inputs
        )

    record_daily_skew(ticker, as_of, rr25)
    history = load_skew_history(ticker, as_of)
    raw_inputs["history_n_obs"] = len(history)

    if len(history) < config.SKEW_MIN_HISTORY_OBS:
        raw_inputs["reason"] = "insufficient skew history"
        return FactorResult(
            factor_name="skew", ticker=ticker, as_of=as_of, score=0.0, raw_inputs=raw_inputs
        )

    mean_rr25 = statistics.fmean(history)
    stdev_rr25 = statistics.pstdev(history)
    raw_inputs["history_mean_rr25"] = mean_rr25
    raw_inputs["history_stdev_rr25"] = stdev_rr25

    if stdev_rr25 == 0:
        score = 0.0
    else:
        z = (rr25 - mean_rr25) / stdev_rr25
        raw_inputs["zscore"] = z
        # More negative RR25 than the norm (z < 0) => richer put skew => positive score.
        score = clip_score(-z / _ZSCORE_SCALE)

    return FactorResult(
        factor_name="skew", ticker=ticker, as_of=as_of, score=score, raw_inputs=raw_inputs
    )
