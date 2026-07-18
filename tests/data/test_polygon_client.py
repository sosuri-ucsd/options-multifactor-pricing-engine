from datetime import date
from unittest.mock import MagicMock

import pytest

from data import polygon_client


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")


def _fake_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def test_list_option_contracts_single_page(monkeypatch):
    monkeypatch.setattr(
        polygon_client.requests,
        "get",
        lambda url, params, timeout: _fake_response(
            {"results": [{"ticker": "O:AAPL240119C00150000"}]}
        ),
    )

    contracts = polygon_client.list_option_contracts("AAPL", date(2026, 1, 1))

    assert contracts == [{"ticker": "O:AAPL240119C00150000"}]


def test_list_option_contracts_paginates(monkeypatch):
    pages = [
        {"results": [{"ticker": "A"}], "next_url": "https://api.polygon.io/next"},
        {"results": [{"ticker": "B"}]},
    ]
    call_count = {"n": 0}

    def fake_get(url, params, timeout):
        page = pages[call_count["n"]]
        call_count["n"] += 1
        return _fake_response(page)

    monkeypatch.setattr(polygon_client.requests, "get", fake_get)

    contracts = polygon_client.list_option_contracts("AAPL", date(2026, 1, 1))

    assert contracts == [{"ticker": "A"}, {"ticker": "B"}]
    assert call_count["n"] == 2


def test_list_option_contracts_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params, timeout):
        calls["n"] += 1
        return _fake_response({"results": [{"ticker": "A"}]})

    monkeypatch.setattr(polygon_client.requests, "get", fake_get)

    polygon_client.list_option_contracts("AAPL", date(2026, 1, 1))
    polygon_client.list_option_contracts("AAPL", date(2026, 1, 1))

    assert calls["n"] == 1


def test_get_contract_daily_bars(monkeypatch):
    monkeypatch.setattr(
        polygon_client.requests,
        "get",
        lambda url, params, timeout: _fake_response(
            {"results": [{"c": 1.23, "t": 1700000000000}]}
        ),
    )

    bars = polygon_client.get_contract_daily_bars(
        "O:AAPL240119C00150000", date(2026, 1, 1), date(2026, 1, 31)
    )

    assert bars == [{"c": 1.23, "t": 1700000000000}]


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        polygon_client.list_option_contracts("AAPL", date(2026, 1, 1))
