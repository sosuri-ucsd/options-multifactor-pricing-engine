"""
Greeks/risk-shape factor: three independent reads on the risk *shape* of a
specific candidate contract, as opposed to whether it's rich or cheap.

1. Pin risk (gamma near expiration). Dollar gamma -- the P&L convexity from
   a 1% move in the underlying -- is:

       dollar_gamma = 0.5 * gamma * S^2 * (0.01)^2

   Gamma mechanically rises as an ATM option approaches expiration, and high
   gamma close to expiry means the position's delta (and a market maker's
   hedging flow) can swing sharply for a small move in the underlying --
   "pin risk". This is scored as a risk factor: we weight dollar gamma more
   heavily the closer the contract is to expiration (exponential weight
   exp(-dte / GREEKS_SHAPE_PIN_RISK_DTE_DECAY)) and score elevated relative
   pin risk (dollar gamma / contract premium) as *unfavorable* -- it adds
   convexity risk that a short-premium position doesn't want.

2. Theta decay rate. For a premium-selling strategy, faster time decay
   relative to the premium collected is favorable -- it's the return being
   harvested. Scored as |theta| / contract_price (daily decay as a fraction
   of premium); higher is better.

3. Delta-profile fit. Covered calls and cash-secured puts both target a
   specific delta band (moderately OTM -- collects meaningful premium
   without a high assignment probability). Fit is scored as how close the
   contract's |delta| sits to GREEKS_SHAPE_TARGET_DELTA, falling off
   linearly to 0 at +-GREEKS_SHAPE_DELTA_TOLERANCE away from the target.

Combined score = 0.4 * theta_component + 0.3 * delta_fit_component
                 + 0.3 * pin_risk_component (already signed negative when risky).
"""
from datetime import date as date_type
from typing import Literal

import config
from factors.base import FactorResult, clip_score

Strategy = Literal["covered_call", "cash_secured_put"]


def dollar_gamma(gamma: float, spot: float) -> float:
    return 0.5 * gamma * spot**2 * 0.01**2


def pin_risk_component(gamma: float, spot: float, contract_price: float, dte: int) -> float:
    if contract_price <= 0:
        return 0.0
    weight = pow(2.718281828, -dte / config.GREEKS_SHAPE_PIN_RISK_DTE_DECAY)
    relative_pin_risk = weight * dollar_gamma(gamma, spot) / contract_price
    return clip_score(-relative_pin_risk / config.GREEKS_SHAPE_PIN_RISK_RELATIVE_SCALE)


def theta_component(theta: float, contract_price: float) -> float:
    if contract_price <= 0:
        return 0.0
    daily_decay_fraction = abs(theta) / contract_price
    return clip_score(daily_decay_fraction / config.GREEKS_SHAPE_THETA_DAILY_SCALE)


def delta_fit_component(delta: float) -> float:
    distance = abs(abs(delta) - config.GREEKS_SHAPE_TARGET_DELTA)
    fit = 1.0 - distance / config.GREEKS_SHAPE_DELTA_TOLERANCE
    return clip_score(fit)


def compute(
    ticker: str,
    as_of: date_type,
    strategy: Strategy,
    delta: float,
    gamma: float,
    theta: float,
    spot: float,
    contract_price: float,
    dte: int,
) -> FactorResult:
    """delta/gamma/theta: per-contract greeks (e.g. from Tradier's chain
    greeks, or computed by pricing/black_scholes.py). `dte`: trading days to
    expiration. `strategy` is recorded for auditability -- the scoring bands
    are currently shared between covered_call and cash_secured_put since
    both target the same moderately-OTM delta profile."""
    pin = pin_risk_component(gamma, spot, contract_price, dte)
    theta_score = theta_component(theta, contract_price)
    delta_fit = delta_fit_component(delta)

    score = 0.4 * theta_score + 0.3 * delta_fit + 0.3 * pin

    raw_inputs = {
        "strategy": strategy,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "spot": spot,
        "contract_price": contract_price,
        "dte": dte,
        "dollar_gamma": dollar_gamma(gamma, spot),
        "pin_risk_component": pin,
        "theta_component": theta_score,
        "delta_fit_component": delta_fit,
    }

    return FactorResult(
        factor_name="greeks_shape", ticker=ticker, as_of=as_of, score=score, raw_inputs=raw_inputs
    )
