"""
Tradier sandbox broker integration: single-leg order submission (covered
calls, cash-secured puts), reconciliation against internally tracked state.

--- Order lifecycle ----------------------------------------------------------

A Tradier order, once POSTed to /accounts/{id}/orders, moves through:

    submitted -> pending -> {open, rejected}
                              open -> {partially_filled, filled, canceled, expired}
                              partially_filled -> {filled, canceled, expired}

"submitted" here just means our POST succeeded and Tradier handed back an
order id -- it does not mean the order has actually reached the market yet.
"pending" is Tradier accepting/validating it internally. "open" means it's
now resting/working. From there it either fills (fully or partially) or
reaches one of the terminal non-fill states: "rejected" (failed validation
or a broker/exchange-level rejection, e.g. insufficient buying power),
"canceled" (we or the system canceled it), or "expired" (a day/GTC order
that ran out its duration unfilled). "filled" and "partially_filled" that
never completes further are also effectively terminal for our purposes.

This project only trades single-leg strategies to start (per section 0/1),
so there's no multi-leg fill-attribution problem yet -- Tradier's
multi-leg ("class": "multileg") order type, where individual legs can fill
at different times, is out of scope until a spread strategy is added.

Reconciliation: Tradier's sandbox doesn't give this pipeline a standing
websocket/streaming connection, so both order status and position state
are checked by polling, not pushed:
  - After submission, poll GET /orders/{id} until status lands in a
    terminal state (poll_until_terminal) -- this is what tells us whether
    an order actually got filled, at what price, rather than trusting that
    submission implies execution.
  - Independently (e.g. once per pipeline run, not per order),
    GET /positions gives the broker's authoritative view of what's
    actually held. reconcile_positions() diffs that against the pipeline's
    own internally tracked position ledger and flags any symbol where they
    disagree -- catching the case where an assignment, a manual account
    action, or an internal bookkeeping bug has caused the two to drift
    apart silently.

--- Risk/liquidity gate -------------------------------------------------------

submit_single_leg_option_order() is the only function in this module that
calls Tradier's order-submission endpoint, and it checks the liquidity gate
(factors/liquidity.py) and the portfolio risk limits (risk/limits.py)
before making that call, returning a blocked OrderResult (never touching
the network) if either fails. There is no other code path in this module
that reaches the network to place an order, so neither gate can be
bypassed by calling something else instead.
"""
import time
from dataclasses import dataclass
from typing import Literal, Optional

import requests

import config
from factors.liquidity import passes_liquidity_gate
from risk.limits import PortfolioExposure, gate_new_order

OrderSide = Literal["sell_to_open", "buy_to_close", "buy_to_open", "sell_to_close"]
OrderType = Literal["market", "limit"]
Duration = Literal["day", "gtc"]

_TERMINAL_STATUSES = {"filled", "rejected", "canceled", "expired"}
_TIMEOUT_SECONDS = 10


def _headers() -> dict:
    api_key = config.require_env(config.ENV_TRADIER_API_KEY)
    return {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}


def _account_id() -> str:
    return config.require_env(config.ENV_TRADIER_ACCOUNT_ID)


@dataclass
class OrderResult:
    order_id: Optional[str]
    status: str  # "submitted" or "blocked"
    blocked_reasons: list[str]


def submit_single_leg_option_order(
    underlying_symbol: str,
    option_symbol: str,
    side: OrderSide,
    quantity: int,
    order_type: OrderType,
    duration: Duration,
    limit_price: Optional[float],
    open_interest: int,
    volume: int,
    bid: float,
    ask: float,
    current_exposure: PortfolioExposure,
    candidate_delta: float,
    candidate_vega: float,
    candidate_beta_weighted_delta: float,
) -> OrderResult:
    if not passes_liquidity_gate(open_interest, volume, bid, ask):
        return OrderResult(order_id=None, status="blocked", blocked_reasons=["failed liquidity gate"])

    gate_result = gate_new_order(
        current_exposure, candidate_delta, candidate_vega, candidate_beta_weighted_delta
    )
    if not gate_result.allowed:
        return OrderResult(order_id=None, status="blocked", blocked_reasons=gate_result.reasons)

    payload = {
        "class": "option",
        "symbol": underlying_symbol,
        "option_symbol": option_symbol,
        "side": side,
        "quantity": quantity,
        "type": order_type,
        "duration": duration,
    }
    if order_type == "limit":
        payload["price"] = limit_price

    resp = requests.post(
        f"{config.TRADIER_SANDBOX_BASE_URL}/accounts/{_account_id()}/orders",
        data=payload,
        headers=_headers(),
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    order_id = str(resp.json()["order"]["id"])
    return OrderResult(order_id=order_id, status="submitted", blocked_reasons=[])


def get_order_status(order_id: str) -> dict:
    resp = requests.get(
        f"{config.TRADIER_SANDBOX_BASE_URL}/accounts/{_account_id()}/orders/{order_id}",
        params={"includeTags": "true"},
        headers=_headers(),
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["order"]


def poll_until_terminal(
    order_id: str, poll_interval_seconds: float = 2.0, timeout_seconds: float = 60.0
) -> dict:
    """Blocks, polling get_order_status, until the order reaches a terminal
    status or timeout_seconds elapses (whichever first) -- the last poll's
    result is returned either way, so a caller can tell an order that timed
    out while still "open" apart from one that actually reached a terminal
    state."""
    elapsed = 0.0
    order = get_order_status(order_id)
    while order.get("status") not in _TERMINAL_STATUSES and elapsed < timeout_seconds:
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds
        order = get_order_status(order_id)
    return order


def get_broker_positions() -> list[dict]:
    resp = requests.get(
        f"{config.TRADIER_SANDBOX_BASE_URL}/accounts/{_account_id()}/positions",
        headers=_headers(),
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    positions = resp.json().get("positions")
    if positions is None:
        return []
    contracts = positions.get("position", [])
    return contracts if isinstance(contracts, list) else [contracts]


@dataclass
class ReconciliationDrift:
    symbol: str
    internal_quantity: float
    broker_quantity: float


def reconcile_positions(internal_positions: dict[str, float]) -> list[ReconciliationDrift]:
    """internal_positions: {symbol: quantity} from the pipeline's own
    tracked ledger. Returns one ReconciliationDrift per symbol where the
    broker's reported quantity disagrees with what's tracked internally --
    including symbols the broker reports that internal state doesn't know
    about at all (internal_quantity defaults to 0), or vice versa."""
    broker_positions = get_broker_positions()
    broker_qty_by_symbol = {p["symbol"]: float(p["quantity"]) for p in broker_positions}

    drifts = []
    for symbol in set(internal_positions) | set(broker_qty_by_symbol):
        internal_qty = internal_positions.get(symbol, 0.0)
        broker_qty = broker_qty_by_symbol.get(symbol, 0.0)
        if internal_qty != broker_qty:
            drifts.append(
                ReconciliationDrift(
                    symbol=symbol, internal_quantity=internal_qty, broker_quantity=broker_qty
                )
            )
    return drifts
