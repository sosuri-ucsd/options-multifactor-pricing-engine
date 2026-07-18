from data import cache


def test_set_and_get_roundtrip():
    cache.set("k1", {"a": 1, "b": [1, 2, 3]})
    assert cache.get("k1") == {"a": 1, "b": [1, 2, 3]}


def test_get_missing_key_returns_none():
    assert cache.get("does-not-exist") is None


def test_ttl_expiry(monkeypatch):
    cache.set("k2", 42)
    assert cache.get("k2", max_age_seconds=1000) == 42

    real_time = cache.time.time

    monkeypatch.setattr(cache.time, "time", lambda: real_time() + 10_000)
    assert cache.get("k2", max_age_seconds=1000) is None


def test_cached_call_only_invokes_fetch_once():
    calls = []

    def fetch():
        calls.append(1)
        return "value"

    key = cache.make_key("src", "AAPL", "2026-01-01")
    first = cache.cached_call(key, fetch)
    second = cache.cached_call(key, fetch)

    assert first == second == "value"
    assert len(calls) == 1


def test_make_key_is_stable_and_readable():
    key = cache.make_key("polygon", "spy", "2026-01-01", expiration="2026-02-01")
    assert key == "polygon:SPY:2026-01-01:2026-02-01"


def test_clear_with_prefix_only_deletes_matching_entries():
    cache.set("polygon:AAPL:2026-01-01", 1)
    cache.set("tradier:AAPL:2026-01-01", 2)

    deleted = cache.clear("polygon:")

    assert deleted == 1
    assert cache.get("polygon:AAPL:2026-01-01") is None
    assert cache.get("tradier:AAPL:2026-01-01") == 2
