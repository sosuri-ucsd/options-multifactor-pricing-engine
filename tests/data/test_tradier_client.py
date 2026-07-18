from unittest.mock import MagicMock

import pytest

from data import tradier_client


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")


def _fake_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def test_get_expirations_normalizes_single_date(monkeypatch):
    monkeypatch.setattr(
        tradier_client.requests,
        "get",
        lambda url, params, headers, timeout: _fake_response(
            {"expirations": {"date": "2026-02-20"}}
        ),
    )

    dates = tradier_client.get_expirations("AAPL")

    assert dates == ["2026-02-20"]


def test_get_expirations_list(monkeypatch):
    monkeypatch.setattr(
        tradier_client.requests,
        "get",
        lambda url, params, headers, timeout: _fake_response(
            {"expirations": {"date": ["2026-02-20", "2026-03-20"]}}
        ),
    )

    dates = tradier_client.get_expirations("AAPL")

    assert dates == ["2026-02-20", "2026-03-20"]


def test_get_chain_with_greeks_normalizes_single_contract(monkeypatch):
    monkeypatch.setattr(
        tradier_client.requests,
        "get",
        lambda url, params, headers, timeout: _fake_response(
            {"options": {"option": {"symbol": "AAPL260220C00150000", "greeks": {"delta": 0.5}}}}
        ),
    )

    chain = tradier_client.get_chain_with_greeks("AAPL", "2026-02-20")

    assert chain == [{"symbol": "AAPL260220C00150000", "greeks": {"delta": 0.5}}]


def test_get_chain_with_greeks_handles_empty_chain(monkeypatch):
    monkeypatch.setattr(
        tradier_client.requests,
        "get",
        lambda url, params, headers, timeout: _fake_response({"options": None}),
    )

    chain = tradier_client.get_chain_with_greeks("AAPL", "2026-02-20")

    assert chain == []


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        tradier_client.get_expirations("AAPL")
