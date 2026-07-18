"""
FRED client: risk-free rate for discounting, read live rather than
hardcoded. Series DGS3MO is the 3-month Treasury constant-maturity yield,
quoted in percent (e.g. 5.31 means 5.31%).
"""
from datetime import date as date_type
from typing import Optional

import requests

import config
from data import cache

_TIMEOUT_SECONDS = 10


def get_risk_free_rate(as_of: Optional[date_type] = None) -> float:
    """Most recent DGS3MO observation on or before `as_of` (default: today),
    returned as a decimal (e.g. 0.0531), not a percent. FRED occasionally has
    no observation for a given day (holidays/weekends), so this asks for a
    short trailing window and takes the latest value in it.
    """
    as_of = as_of or date_type.today()
    key = cache.make_key("fred_dgs3mo", "US", as_of.isoformat())

    def fetch():
        api_key = config.require_env(config.ENV_FRED_API_KEY)
        resp = requests.get(
            config.FRED_BASE_URL,
            params={
                "series_id": config.FRED_RISK_FREE_SERIES,
                "api_key": api_key,
                "file_type": "json",
                "observation_end": as_of.isoformat(),
                "sort_order": "desc",
                "limit": 10,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            if obs["value"] != ".":  # FRED uses "." for missing observations
                return float(obs["value"]) / 100.0
        raise RuntimeError(
            f"No valid {config.FRED_RISK_FREE_SERIES} observation found in the "
            f"10 days before {as_of.isoformat()}"
        )

    max_age = None if as_of < date_type.today() else config.RISK_FREE_CACHE_TTL_SECONDS
    return cache.cached_call(key, fetch, max_age_seconds=max_age)
