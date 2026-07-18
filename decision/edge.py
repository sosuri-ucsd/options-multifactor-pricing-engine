"""
Edge calculation for a single candidate contract.

Both strategies this project starts with -- covered calls and cash-secured
puts -- are short-the-option strategies (you collect a premium and are
short the contract). The spec (section 6) defines "edge" as

    model_price - market_price

which is the edge as a *buyer* of the option would see it: positive means
the option is cheap relative to the model's fair value. That definition is
kept as model_minus_market_edge() for auditability, exactly as specified.

But the P&L that actually accrues to a covered call / cash-secured-put
writer is the opposite sign: you're paid the market price and give up
something the model thinks is worth model_price, so your expected edge is

    edge_per_contract = market_price - model_price

positive when the market is overpaying you (relative to fair value) to
write the option -- exactly the situation a premium seller wants. This is
the number ranking is actually done on; model_minus_market_edge is carried
alongside purely for spec-literal auditability.

Capital required:
  - covered call: you already hold (or are buying) 100 shares per contract,
    so the capital at risk is the stock cost basis, spot * 100.
  - cash-secured put: the position is fully collateralized in cash for
    strike * 100 (no margin), per section 0's "broker-approval-friendly"
    framing for these initial strategies.
"""
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Literal, Optional

from factors.base import FactorResult

CONTRACT_MULTIPLIER = 100
Strategy = Literal["covered_call", "cash_secured_put"]
OptionType = Literal["call", "put"]


@dataclass
class Candidate:
    ticker: str
    strategy: Strategy
    option_type: OptionType
    strike: float
    expiration: date_type
    dte: int
    market_price: float
    model_price: float
    capital_required: float
    factor_results: list[FactorResult] = field(default_factory=list)
    # Everything below is optional and only needed by the execution layer
    # (execution/tradier_broker.py) to actually submit an order for this
    # candidate -- decision/ranking.py never reads these.
    option_symbol: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    delta: Optional[float] = None
    vega: Optional[float] = None


def capital_required_covered_call(spot: float) -> float:
    return spot * CONTRACT_MULTIPLIER


def capital_required_cash_secured_put(strike: float) -> float:
    return strike * CONTRACT_MULTIPLIER


def model_minus_market_edge(candidate: Candidate) -> float:
    """Spec-literal edge definition (section 6): model price minus market
    price, i.e. the edge as seen by a hypothetical buyer of the contract."""
    return candidate.model_price - candidate.market_price


def edge_per_contract(candidate: Candidate) -> float:
    """The edge that actually accrues to a short-premium writer: positive
    when the market pays more than the model's fair value to sell the
    contract away."""
    return candidate.market_price - candidate.model_price


def expected_pnl_per_capital(candidate: Candidate) -> float:
    """Expected P&L (from edge_per_contract, scaled to a full contract) per
    unit of capital/margin the position ties up -- the quantity candidates
    are actually ranked on, not raw edge, since two contracts with identical
    edge but very different collateral requirements are not equally
    attractive uses of capital."""
    if candidate.capital_required <= 0:
        raise ValueError(f"capital_required must be positive, got {candidate.capital_required}")
    return edge_per_contract(candidate) * CONTRACT_MULTIPLIER / candidate.capital_required
