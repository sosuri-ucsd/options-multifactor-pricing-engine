"""
Tradier client: market-data cross-check (vendor-computed IV + Greeks) today,
and the same credentials/account are reused by execution/tradier_broker.py
for paper-trading order placement later.

Uses the sandbox base URL by default (config.TRADIER_SANDBOX_BASE_URL), which
serves both market data and paper trading on one sandbox account -- no need
for separate prod market-data credentials for a research pipeline.
"""
from datetime import date as date_type
from typing import Any

import requests

import config
from data import cache

_TIMEOUT_SECONDS = 10


def _get(path: str, params: dict) -> Any:
    api_key = config.require_env(config.ENV_TRADIER_API_KEY)
    resp = requests.get(
        f"{config.TRADIER_SANDBOX_BASE_URL}{path}",
        params=params,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def get_expirations(symbol: str) -> list[str]:
    """Available option expiration dates (YYYY-MM-DD) for `symbol`."""
    key = cache.make_key("tradier_expirations", symbol, date_type.today().isoformat())

    def fetch():
        payload = _get(
            "/markets/options/expirations",
            {"symbol": symbol, "includeAllRoots": "true"},
        )
        dates = payload.get("expirations", {}).get("date", [])
        return dates if isinstance(dates, list) else [dates]

    return cache.cached_call(key, fetch, max_age_seconds=60 * 60)


def get_chain_with_greeks(symbol: str, expiration: str) -> list[dict]:
    """Full chain for one expiration, including Tradier's vendor-computed
    greeks/IV (requires greeks=true) -- this is the cross-check reference
    used to sanity-check factors/skew.py and pricing/black_scholes.py against
    an independent vol/greeks calculation."""
    key = cache.make_key("tradier_chain", symbol, date_type.today().isoformat(), expiration)

    def fetch():
        payload = _get(
            "/markets/options/chains",
            {"symbol": symbol, "expiration": expiration, "greeks": "true"},
        )
        options = payload.get("options")
        if options is None:
            return []
        contracts = options.get("option", [])
        return contracts if isinstance(contracts, list) else [contracts]

    return cache.cached_call(key, fetch, max_age_seconds=5 * 60)
