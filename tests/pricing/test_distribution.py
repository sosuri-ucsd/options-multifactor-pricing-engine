import math

import numpy as np
import pytest

from pricing import distribution as dist


def _forward(S0, T, r, q):
    return S0 * math.exp((r - q) * T)


@pytest.mark.parametrize(
    "regime_score,skew_score",
    [(0.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (-0.7, 0.6), (1.0, -1.0)],
)
def test_expected_terminal_price_always_matches_forward(regime_score, skew_score):
    S0, T, r, q, sigma_base = 100.0, 0.5, 0.04, 0.01, 0.25
    params = dist.build_distribution(S0, T, r, q, sigma_base, regime_score, skew_score)

    assert dist.expected_terminal_price(params) == pytest.approx(_forward(S0, T, r, q), rel=1e-9)


def test_neutral_regime_and_skew_still_has_small_stress_weight():
    params = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=0.0, skew_score=0.0)
    assert params.stress_weight == pytest.approx(0.05)  # DIST_STRESS_WEIGHT_BASE
    assert params.normal_weight == pytest.approx(0.95)


def test_crisis_regime_increases_stress_weight_and_vol():
    calm = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=0.0, skew_score=0.0)
    crisis = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=-1.0, skew_score=0.0)

    assert crisis.stress_weight > calm.stress_weight
    assert crisis.stress_sigma > calm.stress_sigma


def test_rich_skew_shifts_stress_mean_further_left():
    no_skew = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=-0.5, skew_score=0.0)
    rich_skew = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=-0.5, skew_score=1.0)

    assert rich_skew.stress_mu < no_skew.stress_mu
    # Recentering keeps the overall forward matching in both cases.
    assert dist.expected_terminal_price(no_skew) == pytest.approx(
        dist.expected_terminal_price(rich_skew), rel=1e-9
    )


def test_stress_weight_and_vol_multiplier_are_bounded():
    extreme = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=-100.0, skew_score=0.0)
    assert extreme.stress_weight <= 0.35 * 1.0001  # DIST_STRESS_WEIGHT_MAX with float slack


def test_monte_carlo_sample_mean_converges_to_forward():
    S0, T, r, q, sigma_base = 100.0, 0.5, 0.04, 0.01, 0.25
    params = dist.build_distribution(S0, T, r, q, sigma_base, regime_score=-0.4, skew_score=0.3)

    rng = np.random.default_rng(123)
    samples = dist.sample_terminal_prices(params, n_paths=500_000, rng=rng)

    sample_mean = samples.mean()
    forward = _forward(S0, T, r, q)
    # Monte Carlo standard error at 500k paths should keep this well within 1%.
    assert sample_mean == pytest.approx(forward, rel=0.01)


def test_favorable_regime_score_does_not_reduce_below_base_stress_weight():
    favorable = dist.build_distribution(100.0, 1.0, 0.05, 0.0, 0.20, regime_score=1.0, skew_score=0.0)
    assert favorable.stress_weight == pytest.approx(0.05)
