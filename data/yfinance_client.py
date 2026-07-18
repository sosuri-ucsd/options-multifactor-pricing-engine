"""
yfinance client: underlying daily OHLC and dividend history. Free, no API
key -- used as the default equity-history source since options vendors
typically charge more for equity history than it's worth for this use case.
"""
from datetime import date as date_type

import yfinance as yf

from data import cache


def get_daily_bars(ticker: str, start: date_type, end: date_type) -> list[dict]:
    """Daily OHLCV for `ticker` between start and end (inclusive), used for
    realized-vol calculations (factors/har_rv.py), beta regression
    (factors/beta_regime.py), and momentum (factors/momentum.py)."""
    key = cache.make_key(
        "yfinance_bars", ticker, f"{start.isoformat()}_{end.isoformat()}"
    )

    def fetch():
        history = yf.Ticker(ticker).history(
            start=start.isoformat(), end=end.isoformat(), interval="1d"
        )
        records = []
        for idx, row in history.iterrows():
            records.append(
                {
                    "date": idx.date().isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        return records

    # Only cache indefinitely if the range is fully in the past -- today's bar
    # is still forming intraday.
    max_age = None if end < date_type.today() else 60 * 15
    return cache.cached_call(key, fetch, max_age_seconds=max_age)


def get_dividends(ticker: str) -> list[dict]:
    """Dividend history, used to adjust forward price / carry in the pricing
    distribution for dividend-paying underlyings."""
    key = cache.make_key("yfinance_dividends", ticker, date_type.today().isoformat())

    def fetch():
        divs = yf.Ticker(ticker).dividends
        return [
            {"date": idx.date().isoformat(), "amount": float(amount)}
            for idx, amount in divs.items()
        ]

    return cache.cached_call(key, fetch, max_age_seconds=24 * 60 * 60)
