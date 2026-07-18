"""
Realistic execution costs for backtesting: spread crossing, per-contract
commission, and assignment/exercise handling.

Spread crossing: a resting order at the mid rarely fills; this assumes a
fill BACKTEST_SPREAD_CROSSING_FRACTION of the way across the half-spread,
in the direction that's unfavorable to the trader (buyers pay above mid,
sellers receive below mid) -- 0.5 (fill at the true mid-to-quote midpoint,
i.e. splitting the remaining half-spread) is a reasonable default for
liquid, gated (factors/liquidity.py) contracts; it should be recalibrated
once real fills exist to compare against (execution/'s reconciliation
loop).

Assignment: Tradier and most retail brokers don't charge a separate
exercise/assignment fee (config.BACKTEST_ASSIGNMENT_FEE defaults to 0.0,
but is configurable in case that changes or a different broker is used).
The economic effect of assignment/exercise on P&L (being forced to sell
shares at the call strike, or buy shares at the put strike) is handled by
the caller computing realized P&L from the actual assigned outcome, not by
this module -- this module only models the *fee*, not the P&L mechanics.
"""
from typing import Literal

import config

Side = Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]

_BUY_SIDES = {"buy_to_open", "buy_to_close"}


def fill_price(bid: float, ask: float, side: Side) -> float:
    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0
    crossing = config.BACKTEST_SPREAD_CROSSING_FRACTION * half_spread
    return mid + crossing if side in _BUY_SIDES else mid - crossing


def commission(contracts: int) -> float:
    return contracts * config.BACKTEST_COMMISSION_PER_CONTRACT


def assignment_fee(contracts: int) -> float:
    return contracts * config.BACKTEST_ASSIGNMENT_FEE


def round_trip_cost(contracts: int, entry_bid: float, entry_ask: float, exit_bid: float, exit_ask: float,
                     entry_side: Side, exit_side: Side, assigned: bool = False) -> dict:
    """Total dollar cost (spread crossing + commissions + any assignment fee)
    of opening and closing a position of `contracts` contracts, on top of
    whatever the model priced the position's edge at. Returned as a
    breakdown dict so callers/backtests can attribute cost drag by source."""
    entry_fill = fill_price(entry_bid, entry_ask, entry_side)
    exit_fill = fill_price(exit_bid, exit_ask, exit_side)
    entry_mid = (entry_bid + entry_ask) / 2.0
    exit_mid = (exit_bid + exit_ask) / 2.0

    entry_spread_cost = abs(entry_fill - entry_mid) * contracts * 100
    exit_spread_cost = abs(exit_fill - exit_mid) * contracts * 100
    commission_cost = commission(contracts) * 2  # entry + exit
    assignment_cost = assignment_fee(contracts) if assigned else 0.0

    return {
        "entry_fill_price": entry_fill,
        "exit_fill_price": exit_fill,
        "spread_cost": entry_spread_cost + exit_spread_cost,
        "commission_cost": commission_cost,
        "assignment_cost": assignment_cost,
        "total_cost": entry_spread_cost + exit_spread_cost + commission_cost + assignment_cost,
    }
