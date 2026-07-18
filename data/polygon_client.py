"""
Polygon.io client: options reference data, chain snapshots, and historical
per-contract OHLC aggregates.

Confirm current plan tier / rate limits at https://polygon.io before relying
on high call volumes -- they change, and this client does not itself enforce
a rate limit beyond caching every response.

All network calls go through data.cache.cached_call so repeated backtest
iterations over the same ticker/date/expiration never re-hit the API.
"""
from datetime import date as date_type
from typing import Any, Optional

import requests

import config
from data import cache

_TIMEOUT_SECONDS = 10


def _get(path: str, params: dict) -> Any:
    api_key = config.require_env(config.ENV_POLYGON_API_KEY)
    resp = requests.get(
        f"{config.POLYGON_BASE_URL}{path}",
        params={**params, "apiKey": api_key},
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def list_option_contracts(
    underlying_ticker: str,
    as_of: date_type,
    expired: bool = True,
) -> list[dict]:
    """Reference data: every listed option contract for `underlying_ticker` as of
    `as_of` (strike, expiration, contract type, ticker symbol). expired=True
    includes contracts that have since expired, needed for backtesting."""
    key = cache.make_key(
        "polygon_contracts", underlying_ticker, as_of.isoformat(), suffix=str(expired)
    )

    def fetch():
        results: list[dict] = []
        params = {
            "underlying_ticker": underlying_ticker,
            "as_of": as_of.isoformat(),
            "expired": str(expired).lower(),
            "limit": 1000,
        }
        payload = _get("/v3/reference/options/contracts", params)
        results.extend(payload.get("results", []))
        # Polygon paginates via next_url, which already embeds the api key query
        # param name but not the value -- re-append the key on each hop.
        next_url = payload.get("next_url")
        api_key = config.require_env(config.ENV_POLYGON_API_KEY)
        while next_url:
            resp = requests.get(
                next_url, params={"apiKey": api_key}, timeout=_TIMEOUT_SECONDS
            )
            resp.raise_for_status()
            payload = resp.json()
            results.extend(payload.get("results", []))
            next_url = payload.get("next_url")
        return results

    return cache.cached_call(key, fetch)


def get_chain_snapshot(underlying_ticker: str) -> list[dict]:
    """Live/most-recent chain snapshot: strikes, expirations, quotes, greeks as
    computed by Polygon. Cached with a short TTL since this is a point-in-time
    snapshot, not immutable history."""
    key = cache.make_key("polygon_snapshot", underlying_ticker, date_type.today().isoformat())

    def fetch():
        payload = _get(
            f"/v3/snapshot/options/{underlying_ticker}",
            {"limit": 250},
        )
        return payload.get("results", [])

    return cache.cached_call(key, fetch, max_age_seconds=5 * 60)


def get_contract_daily_bars(
    option_ticker: str,
    start: date_type,
    end: date_type,
) -> list[dict]:
    """Historical daily OHLC for a single option contract ticker (e.g.
    O:AAPL240119C00150000), used for backtesting entry/exit fills. Immutable
    once the date range is in the past, so cached with no TTL."""
    key = cache.make_key(
        "polygon_bars", option_ticker, f"{start.isoformat()}_{end.isoformat()}"
    )

    def fetch():
        payload = _get(
            f"/v2/aggs/ticker/{option_ticker}/range/1/day/"
            f"{start.isoformat()}/{end.isoformat()}",
            {"adjusted": "true", "sort": "asc", "limit": 5000},
        )
        return payload.get("results", [])

    return cache.cached_call(key, fetch)
