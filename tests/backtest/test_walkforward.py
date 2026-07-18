from datetime import date, timedelta

import pytest

from backtest.walkforward import generate_windows


def _dates(n):
    start = date(2024, 1, 1)
    return [start + timedelta(days=i) for i in range(n)]


def test_generate_windows_basic_non_overlapping():
    dates = _dates(30)
    windows = generate_windows(dates, in_sample_days=10, out_sample_days=5)

    # window i is included whenever its out_sample_end index < len(dates);
    # with in=10, out=5, step=5 and 30 dates that's windows starting at
    # index 0, 5, 10, 15 (index 20 would need date index 34, out of range).
    assert len(windows) == 4
    first = windows[0]
    assert first.in_sample_start == dates[0]
    assert first.in_sample_end == dates[9]
    assert first.out_sample_start == dates[10]
    assert first.out_sample_end == dates[14]


def test_generate_windows_rolls_forward_by_step():
    dates = _dates(40)
    windows = generate_windows(dates, in_sample_days=10, out_sample_days=5, step_days=5)

    assert windows[1].in_sample_start == dates[5]
    assert windows[1].out_sample_end == dates[19]


def test_generate_windows_never_uses_future_data_for_in_sample():
    dates = _dates(50)
    windows = generate_windows(dates, in_sample_days=15, out_sample_days=5)
    for w in windows:
        assert w.in_sample_end < w.out_sample_start


def test_generate_windows_empty_when_not_enough_history():
    dates = _dates(10)
    windows = generate_windows(dates, in_sample_days=20, out_sample_days=5)
    assert windows == []


def test_generate_windows_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        generate_windows(_dates(30), in_sample_days=5, out_sample_days=5, step_days=0)
