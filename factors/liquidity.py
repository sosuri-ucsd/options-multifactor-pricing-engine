"""
Liquidity factor -- the one hard gate in the factor set (per the project
spec: liquidity is excluded before it ever reaches the pricing engine,
not just down-weighted).

Three checks, all must pass:
    open_interest >= LIQUIDITY_MIN_OPEN_INTEREST
    volume        >= LIQUIDITY_MIN_VOLUME
    (ask - bid) / mid <= LIQUIDITY_MAX_SPREAD_PCT_OF_MID

These thresholds in config.py are starting points, not calibrated against
real fill data -- tighten them once backtest/execution/reconciliation shows
what spread/OI actually produces fills close to the quoted mid.

A soft 0..1-mapped score is still computed (how far above/inside the
thresholds a contract sits) purely for auditability/tie-breaking among
contracts that already passed the gate -- it never overrides passed_gate.
"""
from datetime import date as date_type
from typing import Optional

import config
from factors.base import FactorResult, clip_score


def spread_pct_of_mid(bid: float, ask: float) -> Optional[float]:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def passes_liquidity_gate(
    open_interest: int, volume: int, bid: float, ask: float
) -> bool:
    spread_pct = spread_pct_of_mid(bid, ask)
    if spread_pct is None:
        return False
    return (
        open_interest >= config.LIQUIDITY_MIN_OPEN_INTEREST
        and volume >= config.LIQUIDITY_MIN_VOLUME
        and spread_pct <= config.LIQUIDITY_MAX_SPREAD_PCT_OF_MID
    )


def _headroom_component(value: float, minimum: float) -> float:
    """0 at the threshold, approaching +1 as value grows to 3x the threshold."""
    if minimum <= 0:
        return 1.0 if value > 0 else -1.0
    return clip_score((value - minimum) / (2 * minimum))


def liquidity_score(open_interest: int, volume: int, bid: float, ask: float) -> float:
    spread_pct = spread_pct_of_mid(bid, ask)
    if spread_pct is None:
        return -1.0

    oi_component = _headroom_component(open_interest, config.LIQUIDITY_MIN_OPEN_INTEREST)
    volume_component = _headroom_component(volume, config.LIQUIDITY_MIN_VOLUME)
    spread_component = clip_score(
        (config.LIQUIDITY_MAX_SPREAD_PCT_OF_MID - spread_pct)
        / config.LIQUIDITY_MAX_SPREAD_PCT_OF_MID
    )
    return (oi_component + volume_component + spread_component) / 3.0


def compute(
    ticker: str,
    as_of: date_type,
    open_interest: int,
    volume: int,
    bid: float,
    ask: float,
) -> FactorResult:
    gate = passes_liquidity_gate(open_interest, volume, bid, ask)
    score = liquidity_score(open_interest, volume, bid, ask)

    raw_inputs = {
        "open_interest": open_interest,
        "volume": volume,
        "bid": bid,
        "ask": ask,
        "spread_pct_of_mid": spread_pct_of_mid(bid, ask),
        "min_open_interest": config.LIQUIDITY_MIN_OPEN_INTEREST,
        "min_volume": config.LIQUIDITY_MIN_VOLUME,
        "max_spread_pct_of_mid": config.LIQUIDITY_MAX_SPREAD_PCT_OF_MID,
    }

    return FactorResult(
        factor_name="liquidity",
        ticker=ticker,
        as_of=as_of,
        score=score,
        raw_inputs=raw_inputs,
        passed_gate=gate,
    )
