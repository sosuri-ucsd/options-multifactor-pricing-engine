from datetime import date
from unittest.mock import MagicMock

import pandas as pd

from data import yfinance_client


def test_get_daily_bars_maps_columns(monkeypatch):
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    history_df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.5, 102.5],
            "Volume": [1_000_000, 1_200_000],
        },
        index=idx,
    )
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = history_df
    monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda t: fake_ticker)

    bars = yfinance_client.get_daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 6))

    assert bars == [
        {
            "date": "2026-01-02",
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.5,
            "volume": 1_000_000.0,
        },
        {
            "date": "2026-01-05",
            "open": 101.0,
            "high": 103.0,
            "low": 100.5,
            "close": 102.5,
            "volume": 1_200_000.0,
        },
    ]


def test_get_daily_bars_is_cached_for_past_ranges(monkeypatch):
    idx = pd.to_datetime(["2026-01-02"])
    history_df = pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]},
        index=idx,
    )
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = history_df
    call_count = {"n": 0}

    def make_ticker(t):
        call_count["n"] += 1
        return fake_ticker

    monkeypatch.setattr(yfinance_client.yf, "Ticker", make_ticker)

    yfinance_client.get_daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 3))
    yfinance_client.get_daily_bars("AAPL", date(2026, 1, 1), date(2026, 1, 3))

    assert call_count["n"] == 1


def test_get_dividends(monkeypatch):
    idx = pd.to_datetime(["2026-02-01"])
    divs = pd.Series([0.24], index=idx)
    fake_ticker = MagicMock()
    fake_ticker.dividends = divs
    monkeypatch.setattr(yfinance_client.yf, "Ticker", lambda t: fake_ticker)

    result = yfinance_client.get_dividends("AAPL")

    assert result == [{"date": "2026-02-01", "amount": 0.24}]
