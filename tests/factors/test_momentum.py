from datetime import date

from factors import momentum


def test_momentum_zscore_none_with_insufficient_history():
    assert momentum.momentum_zscore([100, 101, 102], lookback=20) is None


def test_momentum_zscore_flat_series_none():
    closes = [100.0] * 25
    assert momentum.momentum_zscore(closes, lookback=20) is None  # zero vol -> undefined


def test_momentum_zscore_detects_strong_trend():
    # Small steady daily vol, but a large cumulative move -> high |z|.
    closes = [100.0]
    for i in range(25):
        # alternate tiny noise around a strong upward drift
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1.008))
    z = momentum.momentum_zscore(closes, lookback=20)
    assert z is not None
    assert z > 0


def test_compute_scores_range_bound_market_favorably():
    closes = [100.0]
    for i in range(25):
        closes.append(closes[-1] * (1.001 if i % 2 == 0 else 0.999))
    result = momentum.compute("AAPL", date(2026, 6, 1), closes)
    assert result.score > 0


def test_compute_scores_strong_breakout_unfavorably():
    closes = [100.0]
    for i in range(25):
        # Small alternating wobble so daily-return variance isn't exactly
        # zero (a perfectly constant compounding rate has undefined vol),
        # while still compounding to a strong net uptrend.
        closes.append(closes[-1] * (1.025 if i % 2 == 0 else 1.015))
    result = momentum.compute("AAPL", date(2026, 6, 1), closes)
    assert result.score < 0


def test_compute_neutral_with_insufficient_history():
    result = momentum.compute("AAPL", date(2026, 6, 1), [100.0, 101.0])
    assert result.score == 0.0
