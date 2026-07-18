"""
Local cache for all external market-data pulls.

Options chain history in particular is slow to fetch and metered against
vendor plan limits, so every client in this package (Polygon, Tradier,
yfinance, FRED) must go through this cache rather than re-fetching the same
ticker/date/expiration combination on every backtest iteration.

Storage: a single SQLite file (config.CACHE_DB_PATH). One table, keyed by a
composite string key built from (source, ticker, date, expiration, suffix).
Payloads are stored as JSON text -- this cache is for API responses (chains,
bars, rates), not for large binary blobs, so JSON-in-SQLite is simpler than
mixing in Parquet and is plenty fast at the data volumes a single-account
research pipeline generates.

Entries also record fetched_at so callers can enforce their own TTL (e.g.
FRED risk-free rate should refresh every few hours; historical options bars
for a date that has already closed never need to change and have no TTL).
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    cache_key   TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    fetched_at  REAL NOT NULL
);
"""


def _ensure_db_ready() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect():
    _ensure_db_ready()
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def make_key(
    source: str,
    ticker: str,
    date: str,
    expiration: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """Build a composite cache key. `date` and `expiration` should be ISO strings
    (YYYY-MM-DD) so keys are stable and human-readable for debugging."""
    parts = [source, ticker.upper(), date]
    if expiration:
        parts.append(expiration)
    if suffix:
        parts.append(suffix)
    return ":".join(parts)


def get(cache_key: str, max_age_seconds: Optional[float] = None) -> Optional[Any]:
    """Return the cached payload, or None if missing or older than max_age_seconds.

    max_age_seconds=None means "cached forever" (appropriate for closed
    historical bars); pass a TTL for anything that can change intraday.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM cache_entries WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    payload, fetched_at = row
    if max_age_seconds is not None and (time.time() - fetched_at) > max_age_seconds:
        return None
    return json.loads(payload)


def set(cache_key: str, value: Any) -> None:
    """Store a JSON-serializable payload under cache_key, overwriting any prior entry."""
    payload = json.dumps(value)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO cache_entries (cache_key, payload, fetched_at) VALUES (?, ?, ?)"
            " ON CONFLICT(cache_key) DO UPDATE SET payload = excluded.payload, "
            " fetched_at = excluded.fetched_at",
            (cache_key, payload, time.time()),
        )


def cached_call(
    cache_key: str,
    fetch_fn: Callable[[], Any],
    max_age_seconds: Optional[float] = None,
) -> Any:
    """Return the cached payload for cache_key, calling fetch_fn() and storing the
    result on a miss/expiry. Centralizes the get-or-fetch-then-set pattern used by
    every data client so each one doesn't reimplement it."""
    cached = get(cache_key, max_age_seconds=max_age_seconds)
    if cached is not None:
        return cached
    value = fetch_fn()
    set(cache_key, value)
    return value


def clear(cache_key_prefix: Optional[str] = None) -> int:
    """Delete cache entries. With no prefix, wipes the whole cache. Returns rows deleted."""
    with _connect() as conn:
        if cache_key_prefix is None:
            cur = conn.execute("DELETE FROM cache_entries")
        else:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE cache_key LIKE ?",
                (cache_key_prefix + "%",),
            )
        return cur.rowcount
