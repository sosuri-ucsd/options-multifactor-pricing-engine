from datetime import date

from factors import liquidity


def test_spread_pct_of_mid():
    assert liquidity.spread_pct_of_mid(bid=1.90, ask=2.10) == (2.10 - 1.90) / 2.00


def test_spread_pct_of_mid_zero_mid_returns_none():
    assert liquidity.spread_pct_of_mid(bid=0, ask=0) is None


def test_passes_liquidity_gate_true_when_all_thresholds_met():
    assert liquidity.passes_liquidity_gate(open_interest=500, volume=50, bid=1.95, ask=2.05)


def test_passes_liquidity_gate_false_on_low_open_interest():
    assert not liquidity.passes_liquidity_gate(open_interest=5, volume=50, bid=1.95, ask=2.05)


def test_passes_liquidity_gate_false_on_wide_spread():
    assert not liquidity.passes_liquidity_gate(open_interest=500, volume=50, bid=1.0, ask=2.0)


def test_liquidity_score_negative_on_bad_spread_even_if_gate_data_missing():
    assert liquidity.liquidity_score(0, 0, 0, 0) == -1.0


def test_compute_sets_passed_gate_and_score():
    passing = liquidity.compute("AAPL", date(2026, 6, 1), open_interest=1000, volume=200, bid=1.95, ask=2.05)
    failing = liquidity.compute("AAPL", date(2026, 6, 1), open_interest=1, volume=0, bid=1.0, ask=3.0)

    assert passing.passed_gate is True
    assert failing.passed_gate is False
    assert passing.score > failing.score
