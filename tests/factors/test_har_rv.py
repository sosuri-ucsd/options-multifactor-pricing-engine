import math
from datetime import date, timedelta

import numpy as np

from factors import har_rv


def _synthetic_bars(n, seed=42, daily_vol=0.01, start_price=100.0):
    rng = np.random.default_rng(seed)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * math.exp(rng.normal(0, daily_vol)))
    bars = []
    for c in closes:
        o = c * (1 + rng.normal(0, 0.001))
        h = max(o, c) * (1 + abs(rng.normal(0, 0.002)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.002)))
        bars.append({"open": o, "high": h, "low": l, "close": c})
    return bars


def test_garman_klass_variance_is_nonnegative():
    bars = _synthetic_bars(50)
    variances = har_rv.garman_klass_daily_variance(bars)
    assert len(variances) == 50
    assert all(v >= 0 for v in variances)


def test_garman_klass_zero_range_gives_zero_variance():
    bars = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}]
    assert har_rv.garman_klass_daily_variance(bars) == [0.0]


def test_build_har_features_length_and_shape():
    variances = list(range(1, 31))  # 30 fake daily variances, monotonically increasing
    features = har_rv.build_har_features(variances, weekly_lag_days=5, monthly_lag_days=22)
    assert len(features) == 30 - 22 + 1
    rv_d, rv_w, rv_m = features[0]
    assert rv_d == variances[21]
    assert rv_w == sum(variances[17:22]) / 5
    assert rv_m == sum(variances[0:22]) / 22


def test_fit_har_rv_returns_none_with_insufficient_history():
    variances = har_rv.garman_klass_daily_variance(_synthetic_bars(60))
    assert har_rv.fit_har_rv(variances, horizon_days=10) is None


def test_fit_har_rv_and_forecast_end_to_end():
    bars = _synthetic_bars(400, seed=7)
    variances = har_rv.garman_klass_daily_variance(bars)

    coeffs = har_rv.fit_har_rv(variances, horizon_days=10)

    assert coeffs is not None
    assert len(coeffs) == 4

    forecast = har_rv.forecast_har_rv(coeffs, variances[-1], sum(variances[-5:]) / 5, sum(variances[-22:]) / 22)
    assert forecast >= 0.0


def test_compute_insufficient_history_is_neutral():
    bars = _synthetic_bars(30)
    result = har_rv.compute("TEST", date(2026, 6, 1), bars, current_market_iv=0.25, horizon_days=10)
    assert result.score == 0.0
    assert "reason" in result.raw_inputs


def test_compute_scores_rich_market_iv_positively():
    bars = _synthetic_bars(400, seed=7, daily_vol=0.008)
    # Forecast vol from this low-vol synthetic series will be well under 25%
    # annualized, so a stated market IV far above it should score positive.
    result = har_rv.compute(
        "TEST", date(2026, 6, 1), bars, current_market_iv=0.90, horizon_days=10
    )
    assert result.score > 0
    assert result.raw_inputs["forecast_annualized_vol"] < 0.90


def test_compute_scores_cheap_market_iv_negatively():
    bars = _synthetic_bars(400, seed=7, daily_vol=0.008)
    forecast_vol = har_rv.compute(
        "TEST", date(2026, 6, 1), bars, current_market_iv=None, horizon_days=10
    ).raw_inputs["forecast_annualized_vol"]

    result = har_rv.compute(
        "TEST", date(2026, 6, 1), bars, current_market_iv=forecast_vol * 0.5, horizon_days=10
    )
    assert result.score < 0
