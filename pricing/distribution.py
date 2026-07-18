"""
Custom forecast distribution for the underlying's terminal price, combining
factor outputs instead of trusting the market's implied lognormal.

--- Why a mixture of lognormals, and not the alternatives -------------------

Three standard ways to build a non-lognormal terminal distribution were
considered:

1. Gram-Charlier / Cornish-Fisher expansion on top of a lognormal -- adds
   skewness and excess kurtosis as Hermite-polynomial correction terms to
   the normal density. Analytically clean and used in some classic option
   models (Jarrow-Rudd, Corrado-Su). Rejected here because a truncated
   Gram-Charlier expansion is only a valid probability density for a
   limited range of skew/kurtosis parameters -- push it too far (exactly
   the regime where this factor set wants to express a strong view) and
   the "density" goes negative in the tails. Guarding against that failure
   mode would add real complexity for a personal research pipeline.

2. Merton jump-diffusion -- a compound Poisson jump process overlaid on
   GBM. Theoretically grounded (crash risk really does look like jumps),
   but its parameters (jump intensity, jump-size mean/vol) don't map
   cleanly onto what the factors actually measure -- there's no natural
   translation from "beta_regime factor scored -0.6" to "jump intensity is
   X per year". That indirection makes the model harder to audit, which
   this project cares about (every score must trace back to raw inputs).

3. Two-component mixture of lognormals ("normal" regime + "stress" regime)
   -- the choice made here. A mixture of valid densities is *always* a
   valid density, so there's no failure mode to guard against. It also
   maps directly onto how the factors are already framed: a "regime" read
   (beta_regime) naturally becomes the probability weight on a
   higher-vol/negative-drift stress component, and the model stays fully
   auditable ("20% chance of the stress component, which has 2.4x the base
   vol and a -3% drift shift"). The cost is no closed-form option price --
   but this project is already committed to Monte Carlo pricing specifically
   *because* the distribution won't be a clean lognormal, so that cost is
   already paid regardless of which construction method is chosen.

--- Construction -------------------------------------------------------------

Let X = log(S_T / S_0). Build X as a two-component Gaussian mixture:

    X ~ (1 - w) * N(mu_n, sigma_base^2)      "normal" component
        +   w   * N(mu_s, sigma_stress^2)    "stress" component

Inputs from the factors:
  - sigma_base:    the HAR-RV factor's forecast_annualized_vol -- the vol
                    forecast sets the width of the core component.
  - regime_score:  beta_regime factor score. Its unfavorable direction
                    (negative -- elevated/rising VIX, high beta) increases
                    both the stress weight w and the stress volatility
                    multiplier -- the regime read adjusts tail thickness.
  - skew_score:    skew factor score. Its rich direction (positive -- put
                    skew steeper than this ticker's own norm) pushes the
                    stress component's mean further left -- the skew factor
                    tilts the distribution.

Risk-neutral re-centering: after building the shape above, the mixture's
raw expected value of S_T generally will not equal the no-arbitrage
forward F = S_0 * e^{(r-q)T}. That's fine -- it's supposed to differ in
*shape* from the market's implied lognormal, but not in *drift*, otherwise
"model edge" would really just be an undisclosed directional bet dressed up
as a pricing signal. So both component means are shifted by the same
additive constant c, chosen so the corrected mixture's expected value
matches the forward exactly. A constant shift moves the whole distribution
without changing its relative shape (skew/kurtosis are shift-invariant),
so this recentering doesn't undo the skew/tail work above -- it just
removes an unintended drift artifact from combining the two components.

    E[e^X] (pre-correction) = (1-w) e^{mu_n + 0.5 sigma_base^2}
                                 + w  e^{mu_s + 0.5 sigma_stress^2}
    c = (r - q)T - ln(E[e^X])
    mu_n, mu_s <- mu_n + c, mu_s + c
"""
import math
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class DistributionParams:
    S0: float
    T: float
    r: float
    q: float
    normal_weight: float
    normal_mu: float
    normal_sigma: float
    stress_weight: float
    stress_mu: float
    stress_sigma: float


def _stress_weight(regime_score: float) -> float:
    unfavorable = max(0.0, -regime_score)
    w = config.DIST_STRESS_WEIGHT_BASE + config.DIST_STRESS_WEIGHT_SENSITIVITY * unfavorable
    return min(max(w, config.DIST_STRESS_WEIGHT_MIN), config.DIST_STRESS_WEIGHT_MAX)


def _stress_vol_multiplier(regime_score: float) -> float:
    unfavorable = max(0.0, -regime_score)
    return (
        config.DIST_STRESS_VOL_MULTIPLIER_BASE
        + config.DIST_STRESS_VOL_MULTIPLIER_SENSITIVITY * unfavorable
    )


def build_distribution(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma_base: float,
    regime_score: float,
    skew_score: float,
) -> DistributionParams:
    w = _stress_weight(regime_score)
    stress_sigma = sigma_base * _stress_vol_multiplier(regime_score)

    skew_tilt = max(0.0, skew_score)
    skew_shift = config.DIST_SKEW_SHIFT_SCALE * skew_tilt * math.sqrt(T)

    normal_mu0 = (r - q - 0.5 * sigma_base**2) * T
    stress_mu0 = normal_mu0 - skew_shift

    raw_mean_multiplier = (1 - w) * math.exp(normal_mu0 + 0.5 * sigma_base**2 * T) + w * math.exp(
        stress_mu0 + 0.5 * stress_sigma**2 * T
    )
    correction = (r - q) * T - math.log(raw_mean_multiplier)

    return DistributionParams(
        S0=S0,
        T=T,
        r=r,
        q=q,
        normal_weight=1 - w,
        normal_mu=normal_mu0 + correction,
        normal_sigma=sigma_base * math.sqrt(T),
        stress_weight=w,
        stress_mu=stress_mu0 + correction,
        stress_sigma=stress_sigma * math.sqrt(T),
    )


def sample_terminal_prices(
    params: DistributionParams, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n_paths terminal underlying prices from the constructed mixture."""
    is_stress = rng.random(n_paths) < params.stress_weight
    log_returns = np.where(
        is_stress,
        rng.normal(params.stress_mu, params.stress_sigma, n_paths),
        rng.normal(params.normal_mu, params.normal_sigma, n_paths),
    )
    return params.S0 * np.exp(log_returns)


def build_pure_lognormal(S0: float, T: float, r: float, q: float, sigma: float) -> DistributionParams:
    """A degenerate single-component "mixture" (stress_weight=0) equivalent to
    plain constant-volatility GBM -- i.e. exactly the Black-Scholes
    assumptions, with no skew/tail adjustment. Used to validate the Monte
    Carlo engine itself against the closed-form BSM price before trusting
    the full mixture construction above (pricing/monte_carlo.py's
    validate_against_black_scholes)."""
    mu = (r - q - 0.5 * sigma**2) * T
    sigma_total = sigma * math.sqrt(T)
    return DistributionParams(
        S0=S0,
        T=T,
        r=r,
        q=q,
        normal_weight=1.0,
        normal_mu=mu,
        normal_sigma=sigma_total,
        stress_weight=0.0,
        stress_mu=mu,
        stress_sigma=sigma_total,
    )


def expected_terminal_price(params: DistributionParams) -> float:
    """Analytic E[S_T] under the constructed mixture -- should equal the
    no-arbitrage forward S0 * e^{(r-q)T} by construction; used to validate
    the re-centering correction."""
    normal_term = params.normal_weight * math.exp(
        params.normal_mu + 0.5 * params.normal_sigma**2
    )
    stress_term = params.stress_weight * math.exp(
        params.stress_mu + 0.5 * params.stress_sigma**2
    )
    return params.S0 * (normal_term + stress_term)
