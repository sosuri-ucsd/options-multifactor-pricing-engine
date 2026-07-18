from datetime import date, timedelta

from factors import vol_richness


def _chain(strikes_and_ivs, option_type="call"):
    return [
        {"strike": k, "option_type": option_type, "greeks": {"mid_iv": iv}}
        for k, iv in strikes_and_ivs
    ]


def test_atm_iv_from_chain_picks_nearest_strike():
    chain = _chain([(90, 0.20), (100, 0.25), (110, 0.30)])
    assert vol_richness.atm_iv_from_chain(chain, spot=101) == 0.25


def test_atm_iv_from_chain_empty_returns_none():
    assert vol_richness.atm_iv_from_chain([], spot=100) is None


def test_iv_percentile_rank():
    history = [0.10, 0.15, 0.20, 0.25, 0.30]
    assert vol_richness.iv_percentile_rank(0.20, history) == 60.0  # 3 of 5 <= 0.20
    assert vol_richness.iv_percentile_rank(0.05, history) == 0.0
    assert vol_richness.iv_percentile_rank(0.35, history) == 100.0
    assert vol_richness.iv_percentile_rank(0.5, []) is None


def test_term_structure_slope_signs():
    assert vol_richness.term_structure_slope(near_iv=0.20, far_iv=0.25) > 0  # contango
    assert vol_richness.term_structure_slope(near_iv=0.30, far_iv=0.20) < 0  # backwardation


def test_record_and_load_iv_history_roundtrip():
    today = date(2026, 6, 1)
    vol_richness.record_daily_atm_iv("TEST", today - timedelta(days=2), 0.20)
    vol_richness.record_daily_atm_iv("TEST", today - timedelta(days=1), 0.22)
    vol_richness.record_daily_atm_iv("TEST", today, 0.24)

    history = vol_richness.load_iv_history("TEST", today, lookback_days=30)

    assert history == [0.20, 0.22, 0.24]


def test_load_iv_history_respects_lookback_window():
    today = date(2026, 6, 1)
    vol_richness.record_daily_atm_iv("TEST2", today - timedelta(days=400), 0.10)
    vol_richness.record_daily_atm_iv("TEST2", today, 0.20)

    history = vol_richness.load_iv_history("TEST2", today, lookback_days=30)

    assert history == [0.20]


def test_compute_neutral_when_no_history_yet():
    near = _chain([(100, 0.25)])
    far = _chain([(100, 0.22)])
    result = vol_richness.compute("NEWCO", date(2026, 6, 1), near, far, spot=100)

    # First observation ever -- history has exactly this one point, so
    # percentile rank is trivially 100 (itself is <= itself).
    assert result.factor_name == "vol_richness"
    assert -1.0 <= result.score <= 1.0
    assert result.raw_inputs["near_atm_iv"] == 0.25


def test_compute_scores_rich_iv_positively_after_history_builds():
    ticker = "RICHCO"
    as_of = date(2026, 6, 1)
    # Seed a low-IV history so today's IV ranks at the top (rich).
    for i in range(10, 0, -1):
        vol_richness.record_daily_atm_iv(ticker, as_of - timedelta(days=i), 0.10)

    near = _chain([(100, 0.40)])  # today's IV much higher than history
    far = _chain([(100, 0.40)])   # flat term structure -> isolate the rank effect
    result = vol_richness.compute(ticker, as_of, near, far, spot=100)

    assert result.score > 0
