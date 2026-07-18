"""
Portfolio-level risk limits: net delta, net vega, and beta-weighted net
delta, all configurable in config.py. gate_new_order() is the single
choke point every order must pass through -- it evaluates what the
portfolio's exposures *would become* if the new position were added, and
blocks the order pre-submission if any limit would be breached. Nothing
downstream (execution/) is allowed to submit an order without going
through this gate first.

Beta-weighted net delta expresses every position's delta in SPY-equivalent
terms (position_delta * ticker_beta), so a portfolio of high-beta names
can't quietly carry far more systematic risk than the raw delta limit
alone would catch.
"""
from dataclasses import dataclass

import config


@dataclass
class PortfolioExposure:
    net_delta: float
    net_vega: float
    beta_weighted_net_delta: float


@dataclass
class OrderGateResult:
    allowed: bool
    reasons: list[str]
    resulting_exposure: PortfolioExposure


def gate_new_order(
    current: PortfolioExposure,
    candidate_delta: float,
    candidate_vega: float,
    candidate_beta_weighted_delta: float,
) -> OrderGateResult:
    resulting = PortfolioExposure(
        net_delta=current.net_delta + candidate_delta,
        net_vega=current.net_vega + candidate_vega,
        beta_weighted_net_delta=current.beta_weighted_net_delta + candidate_beta_weighted_delta,
    )

    reasons = []
    if abs(resulting.net_delta) > config.MAX_PORTFOLIO_NET_DELTA:
        reasons.append(
            f"net_delta {resulting.net_delta:.1f} would exceed limit "
            f"{config.MAX_PORTFOLIO_NET_DELTA}"
        )
    if abs(resulting.net_vega) > config.MAX_PORTFOLIO_NET_VEGA:
        reasons.append(
            f"net_vega {resulting.net_vega:.1f} would exceed limit "
            f"{config.MAX_PORTFOLIO_NET_VEGA}"
        )
    if abs(resulting.beta_weighted_net_delta) > config.MAX_PORTFOLIO_BETA_WEIGHTED_DELTA:
        reasons.append(
            f"beta_weighted_net_delta {resulting.beta_weighted_net_delta:.1f} would exceed "
            f"limit {config.MAX_PORTFOLIO_BETA_WEIGHTED_DELTA}"
        )

    return OrderGateResult(allowed=len(reasons) == 0, reasons=reasons, resulting_exposure=resulting)
