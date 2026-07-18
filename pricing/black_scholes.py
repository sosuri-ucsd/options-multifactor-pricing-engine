"""
Black-Scholes-Merton pricer -- the cross-check baseline for every candidate
contract, and the closed-form reference used to validate the Monte Carlo
engine (pricing/monte_carlo.py) before trusting it with a custom
distribution layered on top.

Derivation (standard BSM with continuous dividend yield q):

Assume the underlying follows geometric Brownian motion under the
risk-neutral measure:

    dS = (r - q) S dt + sigma S dW

Then the time-T terminal price is lognormal, and the discounted expected
payoff for a European call/put has the closed form:

    d1 = [ln(S/K) + (r - q + 0.5*sigma^2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    Put  = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)

where N is the standard normal CDF. Greeks follow by differentiating the
price with respect to each input; gamma and vega are identical for calls
and puts because N(d1)'s derivative doesn't depend on which side you're on:

    delta_call =  e^(-qT) * N(d1)
    delta_put  =  e^(-qT) * (N(d1) - 1)
    gamma      =  e^(-qT) * n(d1) / (S * sigma * sqrt(T))
    vega       =  S * e^(-qT) * n(d1) * sqrt(T)              (per 1.00 = 100 vol points)
    theta_call = -S*e^(-qT)*n(d1)*sigma / (2*sqrt(T)) - r*K*e^(-rT)*N(d2)  + q*S*e^(-qT)*N(d1)
    theta_put  = -S*e^(-qT)*n(d1)*sigma / (2*sqrt(T)) + r*K*e^(-rT)*N(-d2) - q*S*e^(-qT)*N(-d1)
    rho_call   =  K * T * e^(-rT) * N(d2)
    rho_put    = -K * T * e^(-rT) * N(-d2)

n(x) is the standard normal PDF. Theta above is per year; this module
returns it divided by 365 (per calendar day) since that's the unit the
rest of the pipeline (greeks_shape factor, risk sizing) reasons in. Vega is
returned per 1 vol point (divided by 100) for the same reason -- "greeks
per 1%" is the market-convention unit, not per 1.00 (100 percentage points).

No dependency on scipy for the normal CDF/PDF -- math.erf gives an exact
closed form for N(x), so pulling in scipy just for this would be overhead.
"""
import math
from typing import Literal

OptionType = Literal["call", "put"]

_SQRT_2PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(
    option_type: OptionType, S: float, K: float, T: float, r: float, q: float, sigma: float
) -> float:
    """S: spot, K: strike, T: years to expiration, r: continuously-compounded
    risk-free rate (decimal), q: continuous dividend yield (decimal),
    sigma: annualized volatility (decimal)."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return S * math.exp(-q * T) * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)
    return K * math.exp(-r * T) * normal_cdf(-d2) - S * math.exp(-q * T) * normal_cdf(-d1)


def greeks(
    option_type: OptionType, S: float, K: float, T: float, r: float, q: float, sigma: float
) -> dict[str, float]:
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)
    pdf_d1 = normal_pdf(d1)

    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega_per_100 = S * disc_q * pdf_d1 * math.sqrt(T) / 100.0

    if option_type == "call":
        delta = disc_q * normal_cdf(d1)
        theta_per_year = (
            -S * disc_q * pdf_d1 * sigma / (2 * math.sqrt(T))
            - r * K * disc_r * normal_cdf(d2)
            + q * S * disc_q * normal_cdf(d1)
        )
        rho = K * T * disc_r * normal_cdf(d2) / 100.0
    else:
        delta = disc_q * (normal_cdf(d1) - 1.0)
        theta_per_year = (
            -S * disc_q * pdf_d1 * sigma / (2 * math.sqrt(T))
            + r * K * disc_r * normal_cdf(-d2)
            - q * S * disc_q * normal_cdf(-d1)
        )
        rho = -K * T * disc_r * normal_cdf(-d2) / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega_per_100,
        "theta": theta_per_year / 365.0,
        "rho": rho,
    }


def implied_vol(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    market_price: float,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Solve for sigma such that price(...) == market_price via bisection.
    Bisection (not Newton-Raphson) is used deliberately: BSM price is
    monotonically increasing in sigma, so bisection over a wide bracket
    always converges, whereas Newton's method needs a decent initial guess
    and can diverge for deep ITM/OTM quotes with a poor starting vol."""
    price_lo = price(option_type, S, K, T, r, q, lo) - market_price
    price_hi = price(option_type, S, K, T, r, q, hi) - market_price
    if price_lo > 0 or price_hi < 0:
        raise ValueError(
            f"market_price {market_price} not attainable for sigma in [{lo}, {hi}]"
        )

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        diff = price(option_type, S, K, T, r, q, mid) - market_price
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0
