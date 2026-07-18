import math

import pytest

from pricing import black_scholes as bs


def test_call_put_parity_no_dividends():
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    call = bs.price("call", S, K, T, r, q, sigma)
    put = bs.price("put", S, K, T, r, q, sigma)
    # Put-call parity: C - P = S*e^-qT - K*e^-rT
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-8)


def test_call_put_parity_with_dividends():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.04, 0.02, 0.25
    call = bs.price("call", S, K, T, r, q, sigma)
    put = bs.price("put", S, K, T, r, q, sigma)
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-8)


def test_atm_call_price_matches_known_reference():
    # Well-known textbook case: S=K=100, T=1, r=5%, sigma=20%, no dividends -> ~10.4506
    price = bs.price("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.20)
    assert price == pytest.approx(10.4506, abs=1e-3)


def test_deep_itm_call_approaches_intrinsic_minus_discount():
    price = bs.price("call", 200.0, 50.0, 0.01, 0.05, 0.0, 0.20)
    intrinsic = 200.0 - 50.0 * math.exp(-0.05 * 0.01)
    assert price == pytest.approx(intrinsic, abs=1e-2)


def test_delta_bounds():
    call_delta = bs.greeks("call", 100, 100, 1.0, 0.05, 0.0, 0.20)["delta"]
    put_delta = bs.greeks("put", 100, 100, 1.0, 0.05, 0.0, 0.20)["delta"]
    assert 0.0 < call_delta < 1.0
    assert -1.0 < put_delta < 0.0


def test_gamma_and_vega_identical_for_call_and_put():
    call_greeks = bs.greeks("call", 100, 105, 0.5, 0.03, 0.01, 0.30)
    put_greeks = bs.greeks("put", 100, 105, 0.5, 0.03, 0.01, 0.30)
    assert call_greeks["gamma"] == pytest.approx(put_greeks["gamma"])
    assert call_greeks["vega"] == pytest.approx(put_greeks["vega"])


def test_gamma_positive_and_vega_positive():
    g = bs.greeks("call", 100, 100, 1.0, 0.05, 0.0, 0.20)
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_implied_vol_roundtrip():
    S, K, T, r, q, true_sigma = 100.0, 105.0, 0.75, 0.04, 0.01, 0.27
    market_price = bs.price("call", S, K, T, r, q, true_sigma)

    recovered = bs.implied_vol("call", S, K, T, r, q, market_price)

    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_implied_vol_roundtrip_put():
    S, K, T, r, q, true_sigma = 100.0, 90.0, 0.25, 0.05, 0.0, 0.35
    market_price = bs.price("put", S, K, T, r, q, true_sigma)

    recovered = bs.implied_vol("put", S, K, T, r, q, market_price)

    assert recovered == pytest.approx(true_sigma, abs=1e-4)


def test_implied_vol_out_of_bracket_raises():
    with pytest.raises(ValueError):
        bs.implied_vol("call", 100.0, 100.0, 1.0, 0.05, 0.0, market_price=1000.0)
