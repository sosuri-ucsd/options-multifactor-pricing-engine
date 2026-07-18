"""
Vol-richness factor.

Two independent reads on "is implied vol expensive right now":

1. IV rank/percentile -- where today's at-the-money IV sits relative to its
   own trailing history. This is IV-rank in the retail-options-trader sense:

       iv_percentile = (# of trailing days with IV <= today's IV) / N * 100

   Percentile (not the alternative "IV rank" formula that only looks at
   min/max) is used because it is robust to a single outlier day blowing out
   the range -- a percentile-of-history measure only shifts gradually.

   Neither Polygon nor Tradier hands back a ready-made trailing IV time
   series for a ticker; a chain snapshot only gives you *today's* IV. So
   this module treats IV history as something the pipeline accumulates
   itself over time (record_daily_atm_iv), same as any live trading desk
   builds its own vol history rather than trusting a vendor's canned
   percentile. Until VOL_RICHNESS_LOOKBACK_DAYS worth of history has been
   recorded, the rank component is scored neutral (0) rather than guessed.

2. Term-structure slope -- near-dated ATM IV vs. far-dated ATM IV for the
   same underlying, on the same day. Backwardation (near-term IV priced
   above far-term) usually reflects near-term event risk or stress and
   historically front-month vol tends to be rich relative to fair value in
   that state; contango (the normal state) carries no particular richness
   signal on its own.

Score = 0.7 * iv_rank_component + 0.3 * term_structure_component, each
mapped independently onto [-1, 1] before combining, then clipped again by
FactorResult.
"""
import math
import statistics
from datetime import date as date_type
from typing import Optional

import config
from data import cache
from factors.base import FactorResult, clip_score

_IV_RANK_WEIGHT = 0.7
_TERM_STRUCTURE_WEIGHT = 0.3
_HISTORY_KEY_PREFIX = "iv_history"


def _history_key(ticker: str) -> str:
    return f"{_HISTORY_KEY_PREFIX}:{ticker.upper()}"


def record_daily_atm_iv(ticker: str, as_of: date_type, atm_iv: float) -> None:
    """Append today's ATM IV observation to this ticker's local history.
    Call once per trading day per ticker (e.g. from the daily pipeline run)
    so iv_percentile_rank has something to compare against over time."""
    key = _history_key(ticker)
    history = cache.get(key) or []
    history = [h for h in history if h["date"] != as_of.isoformat()]
    history.append({"date": as_of.isoformat(), "iv": atm_iv})
    history.sort(key=lambda h: h["date"])
    cache.set(key, history)


def load_iv_history(
    ticker: str, as_of: date_type, lookback_days: int = config.VOL_RICHNESS_LOOKBACK_DAYS
) -> list[float]:
    """IV observations for `ticker` in the trailing `lookback_days` before as_of."""
    history = cache.get(_history_key(ticker)) or []
    cutoff = as_of.toordinal() - lookback_days
    return [
        h["iv"]
        for h in history
        if date_type.fromisoformat(h["date"]).toordinal() <= as_of.toordinal()
        and date_type.fromisoformat(h["date"]).toordinal() >= cutoff
    ]


def iv_vol_of_vol(history: list[float], min_obs: int = 10) -> Optional[float]:
    """Annualized standard deviation of day-over-day ATM IV changes, in vol
    points (e.g. 8.0 for the IV bouncing around by +-8 points/year) -- the
    input risk/sizing.py's vol-targeted position sizing needs to know how
    much *this ticker's own* IV moves around, as opposed to the vol level
    itself. Returns None below min_obs -- an annualized stdev from a
    handful of daily diffs is too noisy to size a position against.
    """
    if len(history) < min_obs:
        return None
    daily_diffs_in_points = [(history[i] - history[i - 1]) * 100 for i in range(1, len(history))]
    daily_stdev = statistics.pstdev(daily_diffs_in_points)
    return daily_stdev * math.sqrt(252)


def atm_iv_from_chain(chain: list[dict], spot: float) -> Optional[float]:
    """Find the contract with strike nearest `spot` in a Tradier-style chain
    (each element has "strike" and "greeks") and return its IV. Prefers
    mid_iv (derived from mid of bid/ask IV) over the single-sided smv_vol
    when both are present, since mid_iv is less sensitive to stale one-sided
    quotes."""
    candidates = [c for c in chain if c.get("strike") is not None and c.get("greeks")]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: abs(c["strike"] - spot))
    greeks = nearest["greeks"]
    iv = greeks.get("mid_iv") or greeks.get("smv_vol") or greeks.get("iv")
    return float(iv) if iv else None


def iv_percentile_rank(current_iv: float, history: list[float]) -> Optional[float]:
    """Percentile (0-100) of current_iv within `history`. None if history is empty."""
    if not history:
        return None
    n = len(history)
    count_at_or_below = sum(1 for h in history if h <= current_iv)
    return 100.0 * count_at_or_below / n


def term_structure_slope(near_iv: float, far_iv: float) -> float:
    """Positive = contango (far > near, normal state). Negative = backwardation
    (near > far, near-term vol priced rich relative to far-term)."""
    if near_iv == 0:
        return 0.0
    return (far_iv - near_iv) / near_iv


def compute(
    ticker: str,
    as_of: date_type,
    near_chain: list[dict],
    far_chain: list[dict],
    spot: float,
) -> FactorResult:
    """near_chain / far_chain: Tradier-style chains (data.tradier_client.get_chain_with_greeks
    output) for the nearest and a farther-dated expiration of the same ticker."""
    near_iv = atm_iv_from_chain(near_chain, spot)
    far_iv = atm_iv_from_chain(far_chain, spot)

    raw_inputs: dict = {"spot": spot, "near_atm_iv": near_iv, "far_atm_iv": far_iv}

    if near_iv is not None:
        record_daily_atm_iv(ticker, as_of, near_iv)
        history = load_iv_history(ticker, as_of)
        rank = iv_percentile_rank(near_iv, history)
    else:
        history, rank = [], None

    raw_inputs["iv_history_n_obs"] = len(history)
    raw_inputs["iv_percentile_rank"] = rank

    if rank is None:
        iv_rank_component = 0.0
    else:
        iv_rank_component = clip_score((rank - 50.0) / 50.0)

    if near_iv is not None and far_iv is not None:
        slope = term_structure_slope(near_iv, far_iv)
        raw_inputs["term_structure_slope"] = slope
        # Backwardation (slope < 0) => rich near-term vol => positive score.
        # Scale so a 20% backwardation maps to a full +1.
        term_structure_component = clip_score(-slope / 0.20)
    else:
        raw_inputs["term_structure_slope"] = None
        term_structure_component = 0.0

    score = (
        _IV_RANK_WEIGHT * iv_rank_component
        + _TERM_STRUCTURE_WEIGHT * term_structure_component
    )

    return FactorResult(
        factor_name="vol_richness",
        ticker=ticker,
        as_of=as_of,
        score=score,
        raw_inputs=raw_inputs,
    )
