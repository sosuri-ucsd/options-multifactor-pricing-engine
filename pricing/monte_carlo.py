"""
Monte Carlo pricer: prices a candidate contract by simulating terminal
underlying prices from the constructed distribution (pricing/distribution.py),
computing the payoff on each path, and discounting at the FRED risk-free
rate. Unlike pricing/black_scholes.py, this also yields the *full*
discounted-payoff distribution across paths -- not just its mean -- which
the risk overlay (risk/) needs to compute vega/vol-budget sizing and
position-level P&L distributions, not just a point price.

--- Validating the engine before trusting it --------------------------------

Layering a custom skewed/fat-tailed distribution on top of Monte Carlo is
only meaningful if the simulation itself is unbiased and converged. That is
checked completely separately from the distribution construction: run the
MC engine on pricing.distribution.build_pure_lognormal (stress_weight=0,
i.e. exactly the constant-volatility, no-skew Black-Scholes assumptions)
and compare against the closed-form BSM price from pricing/black_scholes.py.

The correct statistical check is not "close to within some absolute cents"
-- it's whether the closed-form price falls inside the Monte Carlo
estimate's own confidence interval. The discounted payoffs across paths
have some sample mean (the MC price) and sample standard deviation; by the
central limit theorem the standard error of that mean is
std(payoffs) / sqrt(n_paths), shrinking as 1/sqrt(n). A converged, unbiased
estimator should have |mc_price - bs_price| within a few standard errors
essentially all the time -- if it isn't, that's evidence of a bug, not
sampling noise. convergence_report() below runs the same priced contract at
increasing path counts and reports price + standard error at each, so the
1/sqrt(n) shrinkage is directly visible rather than assumed.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import config
from pricing import black_scholes
from pricing.distribution import DistributionParams, build_pure_lognormal, sample_terminal_prices


@dataclass
class MonteCarloResult:
    price: float
    std_error: float
    discounted_payoffs: np.ndarray
    n_paths: int


def _payoff(option_type: str, terminal_prices: np.ndarray, K: float) -> np.ndarray:
    if option_type == "call":
        return np.maximum(terminal_prices - K, 0.0)
    return np.maximum(K - terminal_prices, 0.0)


def price_option(
    option_type: str,
    K: float,
    params: DistributionParams,
    n_paths: int = config.MC_DEFAULT_NUM_PATHS,
    rng: Optional[np.random.Generator] = None,
) -> MonteCarloResult:
    """rng defaults to a fresh, unseeded Generator -- pass an explicit seeded
    Generator (or set config.MC_RANDOM_SEED) for reproducible backtests."""
    rng = rng if rng is not None else np.random.default_rng(config.MC_RANDOM_SEED)
    terminal_prices = sample_terminal_prices(params, n_paths, rng)
    payoffs = _payoff(option_type, terminal_prices, K)
    discounted = payoffs * math.exp(-params.r * params.T)

    price = float(discounted.mean())
    std_error = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return MonteCarloResult(
        price=price, std_error=std_error, discounted_payoffs=discounted, n_paths=n_paths
    )


def convergence_report(
    option_type: str,
    K: float,
    params: DistributionParams,
    path_counts: tuple[int, ...] = config.MC_CONVERGENCE_CHECK_PATH_COUNTS,
    seed: int = 0,
) -> list[dict]:
    """Price the same contract at each path count in `path_counts`, all
    starting from the same seed, so the reported price/std_error sequence
    shows convergence behavior directly rather than just a single point
    estimate."""
    report = []
    for n_paths in path_counts:
        rng = np.random.default_rng(seed)
        result = price_option(option_type, K, params, n_paths=n_paths, rng=rng)
        report.append({"n_paths": n_paths, "price": result.price, "std_error": result.std_error})
    return report


def validate_against_black_scholes(
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int = config.MC_DEFAULT_NUM_PATHS,
    seed: int = 0,
    n_std_errors: float = 4.0,
) -> dict:
    """Prices the same contract two ways under identical constant-vol,
    no-skew assumptions (build_pure_lognormal): once via this Monte Carlo
    engine, once via closed-form BSM. within_tolerance is True when the
    closed-form price falls within n_std_errors of the MC estimate's own
    standard error -- the statistically correct convergence check, not an
    arbitrary absolute tolerance."""
    params = build_pure_lognormal(S0, T, r, q, sigma)
    rng = np.random.default_rng(seed)
    mc_result = price_option(option_type, K, params, n_paths=n_paths, rng=rng)
    bs_price = black_scholes.price(option_type, S0, K, T, r, q, sigma)

    diff = mc_result.price - bs_price
    within_tolerance = abs(diff) <= n_std_errors * mc_result.std_error

    return {
        "mc_price": mc_result.price,
        "bs_price": bs_price,
        "diff": diff,
        "std_error": mc_result.std_error,
        "within_tolerance": within_tolerance,
    }
