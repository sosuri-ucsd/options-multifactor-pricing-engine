from datetime import date
from unittest.mock import MagicMock

import pytest

from data import fred_client


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")


def _fake_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def test_returns_latest_valid_observation_as_decimal(monkeypatch):
    monkeypatch.setattr(
        fred_client.requests,
        "get",
        lambda url, params, timeout: _fake_response(
            {"observations": [{"date": "2026-01-01", "value": "5.31"}]}
        ),
    )

    rate = fred_client.get_risk_free_rate(date(2026, 1, 1))

    assert rate == pytest.approx(0.0531)


def test_skips_missing_observations(monkeypatch):
    monkeypatch.setattr(
        fred_client.requests,
        "get",
        lambda url, params, timeout: _fake_response(
            {
                "observations": [
                    {"date": "2026-01-03", "value": "."},
                    {"date": "2026-01-01", "value": "5.20"},
                ]
            }
        ),
    )

    rate = fred_client.get_risk_free_rate(date(2026, 1, 3))

    assert rate == pytest.approx(0.0520)


def test_raises_if_no_valid_observation_in_window(monkeypatch):
    monkeypatch.setattr(
        fred_client.requests,
        "get",
        lambda url, params, timeout: _fake_response(
            {"observations": [{"date": "2026-01-01", "value": "."}]}
        ),
    )

    with pytest.raises(RuntimeError):
        fred_client.get_risk_free_rate(date(2026, 1, 1))


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        fred_client.get_risk_free_rate(date(2026, 1, 1))
