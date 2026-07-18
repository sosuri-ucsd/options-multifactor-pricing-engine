import pytest

from backtest import costs


def test_fill_price_buy_pays_above_mid():
    price = costs.fill_price(bid=1.90, ask=2.10, side="buy_to_open")
    assert price == pytest.approx(2.00 + 0.5 * 0.10)


def test_fill_price_sell_receives_below_mid():
    price = costs.fill_price(bid=1.90, ask=2.10, side="sell_to_open")
    assert price == pytest.approx(2.00 - 0.5 * 0.10)


def test_commission_scales_with_contracts():
    assert costs.commission(3) == pytest.approx(3 * 0.65)


def test_assignment_fee_default_zero():
    assert costs.assignment_fee(5) == 0.0


def test_round_trip_cost_breakdown():
    breakdown = costs.round_trip_cost(
        contracts=2,
        entry_bid=1.90, entry_ask=2.10,
        exit_bid=0.90, exit_ask=1.10,
        entry_side="sell_to_open", exit_side="buy_to_close",
    )
    assert breakdown["entry_fill_price"] == pytest.approx(1.95)
    assert breakdown["exit_fill_price"] == pytest.approx(1.05)
    assert breakdown["commission_cost"] == pytest.approx(2 * 0.65 * 2)
    assert breakdown["assignment_cost"] == 0.0
    assert breakdown["total_cost"] == pytest.approx(
        breakdown["spread_cost"] + breakdown["commission_cost"]
    )


def test_round_trip_cost_includes_assignment_fee_when_assigned(monkeypatch):
    monkeypatch.setattr(costs.config, "BACKTEST_ASSIGNMENT_FEE", 5.0)
    breakdown = costs.round_trip_cost(
        contracts=1,
        entry_bid=1.90, entry_ask=2.10,
        exit_bid=0.0, exit_ask=0.0,
        entry_side="sell_to_open", exit_side="buy_to_close",
        assigned=True,
    )
    assert breakdown["assignment_cost"] == 5.0
