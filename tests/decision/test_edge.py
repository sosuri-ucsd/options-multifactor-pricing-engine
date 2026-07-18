from datetime import date

import pytest

from decision.edge import (
    Candidate,
    capital_required_cash_secured_put,
    capital_required_covered_call,
    edge_per_contract,
    expected_pnl_per_capital,
    model_minus_market_edge,
)


def _candidate(market_price, model_price, capital_required):
    return Candidate(
        ticker="AAPL",
        strategy="covered_call",
        option_type="call",
        strike=150.0,
        expiration=date(2026, 7, 17),
        dte=20,
        market_price=market_price,
        model_price=model_price,
        capital_required=capital_required,
        factor_results=[],
    )


def test_capital_required_covered_call():
    assert capital_required_covered_call(spot=150.0) == 15_000.0


def test_capital_required_cash_secured_put():
    assert capital_required_cash_secured_put(strike=140.0) == 14_000.0


def test_model_minus_market_edge_sign():
    c = _candidate(market_price=2.00, model_price=2.50, capital_required=15_000.0)
    assert model_minus_market_edge(c) == pytest.approx(0.50)


def test_edge_per_contract_is_opposite_sign_of_model_minus_market():
    c = _candidate(market_price=2.00, model_price=2.50, capital_required=15_000.0)
    assert edge_per_contract(c) == pytest.approx(-0.50)
    assert edge_per_contract(c) == -model_minus_market_edge(c)


def test_expected_pnl_per_capital_positive_when_market_overpays():
    # Market charges more than the model thinks it's worth -> good for a seller.
    c = _candidate(market_price=3.00, model_price=2.00, capital_required=15_000.0)
    pnl_per_capital = expected_pnl_per_capital(c)
    assert pnl_per_capital == pytest.approx(1.00 * 100 / 15_000.0)
    assert pnl_per_capital > 0


def test_expected_pnl_per_capital_negative_when_market_underpays():
    c = _candidate(market_price=1.50, model_price=2.00, capital_required=15_000.0)
    assert expected_pnl_per_capital(c) < 0


def test_expected_pnl_per_capital_raises_on_nonpositive_capital():
    c = _candidate(market_price=2.00, model_price=2.50, capital_required=0.0)
    with pytest.raises(ValueError):
        expected_pnl_per_capital(c)
