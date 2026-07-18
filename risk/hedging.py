"""
Auto delta-hedging: when net delta (position-level or portfolio-level, in
shares-equivalent -- i.e. sum of contract_delta * 100 * num_contracts)
drifts past config.DELTA_HEDGE_TRIGGER_THRESHOLD, generate an offsetting
order in the underlying to bring net delta back to flat.

Hedging back to exactly zero (rather than just back inside the band) is
the simpler and more conservative choice -- it removes directional risk
completely rather than leaving a residual within tolerance, at the cost of
slightly more hedge turnover. That trade-off is worth revisiting once
transaction costs from real fills are measured (backtest/), but starting
conservative is the right default for a system that will eventually place
real trades.
"""
from typing import Optional, TypedDict

import config


class HedgeOrder(TypedDict):
    side: str  # "buy" or "sell"
    shares: int


def needs_delta_hedge(
    net_delta: float, threshold: float = config.DELTA_HEDGE_TRIGGER_THRESHOLD
) -> bool:
    return abs(net_delta) > threshold


def generate_hedge_order(
    net_delta: float, threshold: float = config.DELTA_HEDGE_TRIGGER_THRESHOLD
) -> Optional[HedgeOrder]:
    if not needs_delta_hedge(net_delta, threshold):
        return None
    shares_to_trade = -net_delta  # offset exactly back to zero net delta
    side = "buy" if shares_to_trade > 0 else "sell"
    return {"side": side, "shares": int(round(abs(shares_to_trade)))}
