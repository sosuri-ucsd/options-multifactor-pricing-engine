import pytest

from risk import sizing


def test_kelly_fraction_reference_formula():
    # p=0.6, b=1.5 -> f* = 0.6 - 0.4/1.5
    assert sizing.kelly_fraction(0.6, 1.5) == pytest.approx(0.6 - 0.4 / 1.5)


def test_contract_vega_dollar_vol():
    assert sizing.contract_vega_dollar_vol(vega_per_contract=50.0, iv_annual_vol_of_vol_points=8.0) == 400.0
    # Sign of vega shouldn't matter -- it's a magnitude of risk either way.
    assert sizing.contract_vega_dollar_vol(vega_per_contract=-50.0, iv_annual_vol_of_vol_points=8.0) == 400.0


def test_max_contracts_within_vol_budget_basic():
    # 400 dollars of vol per contract, 2000 dollar budget remaining -> 5 contracts
    n = sizing.max_contracts_within_vol_budget(
        vega_per_contract=50.0, iv_annual_vol_of_vol_points=8.0, remaining_vol_budget_dollars=2000.0
    )
    assert n == 5


def test_max_contracts_within_vol_budget_zero_remaining():
    n = sizing.max_contracts_within_vol_budget(50.0, 8.0, remaining_vol_budget_dollars=0.0)
    assert n == 0


def test_max_contracts_within_vol_budget_negative_remaining():
    n = sizing.max_contracts_within_vol_budget(50.0, 8.0, remaining_vol_budget_dollars=-100.0)
    assert n == 0


def test_max_contracts_zero_vega_is_unconstrained():
    n = sizing.max_contracts_within_vol_budget(0.0, 8.0, remaining_vol_budget_dollars=100.0)
    assert n > 10**6


def test_vol_target_position_size_end_to_end():
    n = sizing.vol_target_position_size(
        account_capital=1_000_000.0,
        existing_portfolio_vol_used_dollars=0.0,
        vega_per_contract=50.0,
        iv_annual_vol_of_vol_points=8.0,
        vol_budget_fraction=0.15,
    )
    # budget = 150,000; per-contract vol = 400 -> 375 contracts
    assert n == 375


def test_vol_target_position_size_shrinks_as_existing_usage_grows():
    fresh = sizing.vol_target_position_size(1_000_000.0, 0.0, 50.0, 8.0, 0.15)
    used_up = sizing.vol_target_position_size(1_000_000.0, 100_000.0, 50.0, 8.0, 0.15)
    assert used_up < fresh
