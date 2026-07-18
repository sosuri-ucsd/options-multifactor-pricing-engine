from datetime import date, timedelta

from factors import skew


def _chain():
    return [
        {"option_type": "call", "greeks": {"delta": 0.25, "mid_iv": 0.22}},
        {"option_type": "call", "greeks": {"delta": 0.50, "mid_iv": 0.20}},
        {"option_type": "put", "greeks": {"delta": -0.25, "mid_iv": 0.28}},
        {"option_type": "put", "greeks": {"delta": -0.50, "mid_iv": 0.21}},
    ]


def test_risk_reversal_25d_picks_correct_contracts():
    rr25 = skew.risk_reversal_25d(_chain())
    assert rr25 == 0.22 - 0.28  # call_iv - put_iv, negative equity put-skew


def test_risk_reversal_25d_missing_greeks_returns_none():
    chain = [{"option_type": "call", "greeks": {}}]
    assert skew.risk_reversal_25d(chain) is None


def test_record_and_load_skew_history_roundtrip():
    today = date(2026, 6, 1)
    skew.record_daily_skew("SKEWCO", today - timedelta(days=1), -0.05)
    skew.record_daily_skew("SKEWCO", today, -0.06)

    history = skew.load_skew_history("SKEWCO", today, lookback_days=30)

    assert history == [-0.05, -0.06]


def test_compute_neutral_with_insufficient_history():
    result = skew.compute("NEWCO2", date(2026, 6, 1), _chain())
    assert result.score == 0.0
    assert "reason" in result.raw_inputs


def test_compute_scores_steepening_skew_positively():
    ticker = "STEEPCO"
    as_of = date(2026, 6, 1)
    # Seed a mild historical skew with a little day-to-day variation so the
    # stdev is nonzero (a perfectly flat history would make the z-score
    # undefined and the factor falls back to neutral by design).
    for i in range(25, 0, -1):
        wobble = 0.002 if i % 2 == 0 else -0.002
        skew.record_daily_skew(ticker, as_of - timedelta(days=i), -0.02 + wobble)

    # Today's skew is far more negative (put IV much richer than usual).
    chain = [
        {"option_type": "call", "greeks": {"delta": 0.25, "mid_iv": 0.20}},
        {"option_type": "put", "greeks": {"delta": -0.25, "mid_iv": 0.35}},
    ]
    result = skew.compute(ticker, as_of, chain)

    assert result.score > 0
    assert result.raw_inputs["zscore"] < 0
