"""
Walk-forward windowing: a rolling in-sample fit window followed by an
out-of-sample test window, rolled forward across the full available
history -- never a single fixed train/test split, which would only tell
you whether the strategy worked in one particular historical regime.

Given a chronological list of trading-day dates, generate_windows() yields
(in_sample, out_sample) date-range pairs, advancing by `step_days` each
time (defaulting to the out-of-sample length, i.e. non-overlapping
out-of-sample periods -- every trading day is tested exactly once, using
only information available before it).
"""
from dataclasses import dataclass
from datetime import date as date_type
from typing import Optional


@dataclass
class WalkForwardWindow:
    in_sample_start: date_type
    in_sample_end: date_type
    out_sample_start: date_type
    out_sample_end: date_type


def generate_windows(
    all_dates: list[date_type],
    in_sample_days: int,
    out_sample_days: int,
    step_days: Optional[int] = None,
) -> list[WalkForwardWindow]:
    """all_dates must be sorted ascending. Windows are built purely by index
    position within all_dates (i.e. in trading days, not calendar days), so
    weekends/holidays that are simply absent from all_dates don't distort
    window lengths."""
    step = step_days if step_days is not None else out_sample_days
    if step <= 0:
        raise ValueError("step_days must be positive")

    windows = []
    n = len(all_dates)
    i = 0
    while True:
        in_end_idx = i + in_sample_days - 1
        out_start_idx = in_end_idx + 1
        out_end_idx = out_start_idx + out_sample_days - 1
        if out_end_idx >= n:
            break
        windows.append(
            WalkForwardWindow(
                in_sample_start=all_dates[i],
                in_sample_end=all_dates[in_end_idx],
                out_sample_start=all_dates[out_start_idx],
                out_sample_end=all_dates[out_end_idx],
            )
        )
        i += step
    return windows
