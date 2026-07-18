from datetime import date

from factors import greeks_shape


def test_dollar_gamma_formula():
    assert greeks_shape.dollar_gamma(gamma=0.05, spot=100) == 0.5 * 0.05 * 100**2 * 0.01**2


def test_pin_risk_component_worse_close_to_expiration():
    far_dte_score = greeks_shape.pin_risk_component(
        gamma=0.10, spot=100, contract_price=2.0, dte=30
    )
    near_dte_score = greeks_shape.pin_risk_component(
        gamma=0.10, spot=100, contract_price=2.0, dte=1
    )
    assert near_dte_score < far_dte_score


def test_pin_risk_component_zero_price_is_neutral():
    assert greeks_shape.pin_risk_component(gamma=0.1, spot=100, contract_price=0, dte=5) == 0.0


def test_theta_component_scales_with_decay_fraction():
    low_decay = greeks_shape.theta_component(theta=-0.01, contract_price=2.0)
    high_decay = greeks_shape.theta_component(theta=-0.10, contract_price=2.0)
    assert high_decay > low_decay


def test_delta_fit_component_peaks_at_target():
    at_target = greeks_shape.delta_fit_component(0.30)
    away_from_target = greeks_shape.delta_fit_component(0.60)
    assert at_target == 1.0
    assert away_from_target < at_target


def test_compute_combines_components():
    result = greeks_shape.compute(
        "AAPL",
        date(2026, 6, 1),
        strategy="covered_call",
        delta=0.30,
        gamma=0.02,
        theta=-0.05,
        spot=150.0,
        contract_price=2.5,
        dte=20,
    )
    assert result.factor_name == "greeks_shape"
    assert -1.0 <= result.score <= 1.0
    assert result.raw_inputs["delta_fit_component"] == 1.0
