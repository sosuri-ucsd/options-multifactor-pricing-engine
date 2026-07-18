"""
Position sizing: vol-targeting.

--- Why vol-targeting instead of (fractional) Kelly -------------------------

The Kelly criterion sizes a bet as a fraction of capital:

    f* = p - (1 - p) / b

where p is the win probability and b is the win/loss payoff ratio. For a
premium-selling strategy, both p and b would have to come from this same
pricing engine's own edge estimate -- there's no independent source for
them. That's the problem: Kelly sizing is well known to be extremely
sensitive to mis-estimating p and b, and early in this project's life
(before there's a live track record to calibrate against) the factor
weights, distribution construction, and MC pricing are exactly the
components most likely to have real estimation error. Sizing bets as a
direct function of a possibly-wrong edge estimate means a model bug or a
bad factor calibration doesn't just produce a bad trade -- it produces an
oversized bad trade. kelly_fraction() below is still provided as a
reference/comparison calculation (and a natural input to a *secondary*
scaling factor later, once there's enough live history to estimate p and b
independently of the model), but it is not what position sizes are based
on today.

Vol-targeting sizes independently of the edge estimate's magnitude: it asks
"how much portfolio volatility am I willing to carry from this position",
not "how good does the model think this trade is". That makes position
size robust to the pricing model being wrong, which matters more for a
system that "will eventually place real trades" than capturing the last
bit of theoretical edge would.

--- Derivation ---------------------------------------------------------------

A position's dollar vega (per contract) is the P&L sensitivity to a 1-vol-
point change in implied vol: pricing.black_scholes.greeks()["vega"] is
already scaled per 1 point per share, so per-contract dollar vega is
vega_per_share * 100.

If implied vol itself moves over a year with some annualized standard
deviation sigma_iv (in vol points -- e.g. IV bouncing around by 8 points a
year), then the annualized dollar P&L standard deviation this position
contributes purely from vega exposure is approximately:

    position_vol_contribution = |vega_per_contract| * sigma_iv

This is a first-order (delta-in-vega) approximation -- it ignores the
vega-of-vega and any correlation structure between this position's
underlying's IV and everything else already in the book. On that last
point: rather than assuming vega P&L across different positions is
uncorrelated (and summing variances), this budget sums vol contributions
*linearly* across positions. That's deliberately conservative -- implied
vols across equities are typically highly correlated during the exact
regime (a broad vol spike) where this risk actually shows up, so treating
them as if they move together is more realistic than assuming
diversification that likely evaporates when it would matter most.

Sizing: given a total portfolio vol budget (config.PORTFOLIO_VOL_BUDGET_ANNUAL
as a fraction of account capital) and however much of that budget existing
positions already use, the number of new contracts is capped at whatever
keeps the *cumulative* vol contribution within budget:

    remaining_budget = vol_budget_fraction * account_capital - existing_portfolio_vol_used
    max_contracts     = floor(remaining_budget / (|vega_per_contract| * sigma_iv))
"""
import config

# Sentinel returned when a position has ~zero vega and so is unconstrained by
# the vol budget -- an int (not float("inf")) so callers can keep treating
# the return value as a plain contract count.
_UNCONSTRAINED_MAX_CONTRACTS = 10**9


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Reference/comparison only -- see module docstring for why this isn't
    used to size positions today. f* = p - (1-p)/b."""
    return win_prob - (1 - win_prob) / win_loss_ratio


def contract_vega_dollar_vol(vega_per_contract: float, iv_annual_vol_of_vol_points: float) -> float:
    """Annualized dollar P&L standard deviation one contract contributes from
    vega exposure, given an estimate of how much this underlying's IV itself
    moves around per year (in vol points, e.g. 8.0 for +-8 points/year)."""
    return abs(vega_per_contract) * iv_annual_vol_of_vol_points


def max_contracts_within_vol_budget(
    vega_per_contract: float,
    iv_annual_vol_of_vol_points: float,
    remaining_vol_budget_dollars: float,
) -> int:
    per_contract_vol = contract_vega_dollar_vol(vega_per_contract, iv_annual_vol_of_vol_points)
    if per_contract_vol <= 0:
        # Zero vega (e.g. an already-expired or deep ITM/OTM contract with no
        # residual optionality) contributes nothing to the vol budget, so it
        # isn't constrained by it -- sizing should be bounded elsewhere
        # (portfolio delta/beta limits in risk/limits.py) instead.
        return _UNCONSTRAINED_MAX_CONTRACTS
    if remaining_vol_budget_dollars <= 0:
        return 0
    return int(remaining_vol_budget_dollars // per_contract_vol)


def vol_target_position_size(
    account_capital: float,
    existing_portfolio_vol_used_dollars: float,
    vega_per_contract: float,
    iv_annual_vol_of_vol_points: float,
    vol_budget_fraction: float = config.PORTFOLIO_VOL_BUDGET_ANNUAL,
) -> int:
    """Maximum number of contracts of this specific position the portfolio
    vol budget allows, given how much of the budget existing positions
    already use."""
    total_budget = vol_budget_fraction * account_capital
    remaining = total_budget - existing_portfolio_vol_used_dollars
    return max_contracts_within_vol_budget(
        vega_per_contract, iv_annual_vol_of_vol_points, remaining
    )
