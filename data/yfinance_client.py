"""
yfinance client: underlying daily OHLC and dividend history. Free, no API
key -- used as the default equity-history source since options vendors
typically charge more for equity history than it's worth for this use case.
"""
import math
from datetime import date as date_type, timedelta

import yfinance as yf

from data import cache

DIVIDEND_LOOKBACK_DAYS = 365


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


def estimate_dividend_yield(
    ticker: str, as_of: date_type, spot: float, lookback_days: int = DIVIDEND_LOOKBACK_DAYS
) -> float:
    """Forward continuous dividend yield q, estimated from trailing dividends
    -- there's no forward-looking dividend forecast source wired in, so this
    assumes next year's payments will resemble the trailing lookback_days
    (a standard simplification for a name with a stable payout policy; it
    will understate/overstate yield around an actual dividend change).

    Trailing dividends / spot gives a simple annualized yield. That's
    converted to the continuous-compounding rate q used everywhere else in
    the pricing engine (S * e^(-qT)) via q = ln(1 + simple_yield), rather
    than using the simple yield directly -- the two are close for typical
    equity yields (0-5%) but this keeps the convention exact.
    """
    if spot <= 0:
        return 0.0
    dividends = get_dividends(ticker)
    cutoff = as_of - timedelta(days=lookback_days)
    trailing_total = sum(
        d["amount"]
        for d in dividends
        if cutoff <= date_type.fromisoformat(d["date"]) <= as_of
    )
    if trailing_total <= 0:
        return 0.0
    simple_yield = trailing_total / spot
    return math.log(1 + simple_yield)
