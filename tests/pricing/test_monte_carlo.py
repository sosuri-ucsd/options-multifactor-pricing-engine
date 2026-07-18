import numpy as np
import pytest

from pricing import black_scholes, monte_carlo
from pricing.distribution import build_distribution, build_pure_lognormal


@pytest.mark.parametrize(
    "option_type,S0,K,T,r,q,sigma",
    [
        ("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20),   # ATM
        ("put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20),    # ATM
        ("call", 100.0, 120.0, 0.5, 0.03, 0.01, 0.35),  # OTM call
        ("put", 100.0, 80.0, 0.25, 0.04, 0.0, 0.45),    # OTM put, high vol, short-dated
    ],
)
def test_mc_matches_black_scholes_within_confidence_interval(option_type, S0, K, T, r, q, sigma):
    result = monte_carlo.validate_against_black_scholes(
        option_type, S0, K, T, r, q, sigma, n_paths=200_000, seed=42
    )
    assert result["within_tolerance"], result


def test_convergence_report_std_error_shrinks_with_more_paths():
    params = build_pure_lognormal(100.0, 1.0, 0.05, 0.0, 0.20)
    report = monte_carlo.convergence_report("call", 100.0, params, path_counts=(1_000, 10_000, 100_000))

    std_errors = [row["std_error"] for row in report]
    assert std_errors[0] > std_errors[1] > std_errors[2]


def test_convergence_report_prices_stabilize_toward_bs_price():
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    params = build_pure_lognormal(S0, T, r, q, sigma)
    bs_price = black_scholes.price("call", S0, K, T, r, q, sigma)

    report = monte_carlo.convergence_report(
        "call", K, params, path_counts=(1_000, 500_000), seed=7
    )

    error_small_n = abs(report[0]["price"] - bs_price)
    error_large_n = abs(report[-1]["price"] - bs_price)
    assert error_large_n < error_small_n


def test_price_option_returns_full_payoff_distribution():
    params = build_pure_lognormal(100.0, 1.0, 0.05, 0.0, 0.20)
    rng = np.random.default_rng(1)
    result = monte_carlo.price_option("call", 100.0, params, n_paths=5_000, rng=rng)

    assert result.n_paths == 5_000
    assert len(result.discounted_payoffs) == 5_000
    assert result.price >= 0
    assert (result.discounted_payoffs >= 0).all()


def test_mixture_distribution_price_differs_from_pure_lognormal_under_stress():
    S0, K, T, r, q, sigma = 100.0, 90.0, 0.5, 0.04, 0.0, 0.20

    calm_params = build_pure_lognormal(S0, T, r, q, sigma)
    stressed_params = build_distribution(S0, T, r, q, sigma, regime_score=-1.0, skew_score=1.0)

    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    calm_put_price = monte_carlo.price_option("put", K, calm_params, n_paths=300_000, rng=rng1).price
    stressed_put_price = monte_carlo.price_option(
        "put", K, stressed_params, n_paths=300_000, rng=rng2
    ).price

    # Fatter left tail + negative skew shift should make an OTM put worth more.
    assert stressed_put_price > calm_put_price
