from unittest.mock import MagicMock

import pytest

from execution import tradier_broker as broker
from risk.limits import PortfolioExposure


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "test-key")
    monkeypatch.setenv("TRADIER_ACCOUNT_ID", "test-account")


def _fake_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status.return_value = None
    return resp


def _flat_exposure():
    return PortfolioExposure(net_delta=0.0, net_vega=0.0, beta_weighted_net_delta=0.0)


def test_order_blocked_by_liquidity_gate_never_calls_network(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(broker.requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    result = broker.submit_single_leg_option_order(
        underlying_symbol="AAPL",
        option_symbol="AAPL260220P00150000",
        side="sell_to_open",
        quantity=1,
        order_type="limit",
        duration="day",
        limit_price=2.00,
        open_interest=1,  # fails liquidity gate
        volume=0,
        bid=1.0,
        ask=3.0,
        current_exposure=_flat_exposure(),
        candidate_delta=10,
        candidate_vega=10,
        candidate_beta_weighted_delta=10,
    )

    assert result.status == "blocked"
    assert "liquidity" in result.blocked_reasons[0]
    assert calls["n"] == 0


def test_order_blocked_by_risk_limits_never_calls_network(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(broker.requests, "post", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    result = broker.submit_single_leg_option_order(
        underlying_symbol="AAPL",
        option_symbol="AAPL260220P00150000",
        side="sell_to_open",
        quantity=1,
        order_type="limit",
        duration="day",
        limit_price=2.00,
        open_interest=1000,
        volume=500,
        bid=1.95,
        ask=2.05,
        current_exposure=_flat_exposure(),
        candidate_delta=10_000,  # breaches MAX_PORTFOLIO_NET_DELTA
        candidate_vega=0,
        candidate_beta_weighted_delta=0,
    )

    assert result.status == "blocked"
    assert any("net_delta" in r for r in result.blocked_reasons)
    assert calls["n"] == 0


def test_order_submitted_when_gates_pass(monkeypatch):
    monkeypatch.setattr(
        broker.requests,
        "post",
        lambda url, data, headers, timeout: _fake_response({"order": {"id": 12345, "status": "ok"}}),
    )

    result = broker.submit_single_leg_option_order(
        underlying_symbol="AAPL",
        option_symbol="AAPL260220P00150000",
        side="sell_to_open",
        quantity=1,
        order_type="limit",
        duration="day",
        limit_price=2.00,
        open_interest=1000,
        volume=500,
        bid=1.95,
        ask=2.05,
        current_exposure=_flat_exposure(),
        candidate_delta=10,
        candidate_vega=10,
        candidate_beta_weighted_delta=10,
    )

    assert result.status == "submitted"
    assert result.order_id == "12345"


def test_get_order_status(monkeypatch):
    monkeypatch.setattr(
        broker.requests,
        "get",
        lambda url, params=None, headers=None, timeout=None: _fake_response(
            {"order": {"id": 1, "status": "filled"}}
        ),
    )
    order = broker.get_order_status("1")
    assert order["status"] == "filled"


def test_poll_until_terminal_stops_at_first_terminal_status(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        status = "open" if calls["n"] < 3 else "filled"
        return _fake_response({"order": {"id": 1, "status": status}})

    monkeypatch.setattr(broker.requests, "get", fake_get)
    monkeypatch.setattr(broker.time, "sleep", lambda s: None)

    order = broker.poll_until_terminal("1", poll_interval_seconds=0.01, timeout_seconds=5)

    assert order["status"] == "filled"
    assert calls["n"] == 3


def test_poll_until_terminal_gives_up_after_timeout(monkeypatch):
    monkeypatch.setattr(
        broker.requests,
        "get",
        lambda url, params=None, headers=None, timeout=None: _fake_response(
            {"order": {"id": 1, "status": "open"}}
        ),
    )
    monkeypatch.setattr(broker.time, "sleep", lambda s: None)

    order = broker.poll_until_terminal("1", poll_interval_seconds=1, timeout_seconds=2)

    assert order["status"] == "open"


def test_get_broker_positions_normalizes_single_position(monkeypatch):
    monkeypatch.setattr(
        broker.requests,
        "get",
        lambda url, headers=None, timeout=None: _fake_response(
            {"positions": {"position": {"symbol": "AAPL260220P00150000", "quantity": -1}}}
        ),
    )
    positions = broker.get_broker_positions()
    assert positions == [{"symbol": "AAPL260220P00150000", "quantity": -1}]


def test_get_broker_positions_handles_no_positions(monkeypatch):
    monkeypatch.setattr(
        broker.requests,
        "get",
        lambda url, headers=None, timeout=None: _fake_response({"positions": None}),
    )
    assert broker.get_broker_positions() == []


def test_reconcile_positions_flags_drift(monkeypatch):
    monkeypatch.setattr(
        broker,
        "get_broker_positions",
        lambda: [{"symbol": "AAPL260220P00150000", "quantity": -2}],
    )

    drifts = broker.reconcile_positions({"AAPL260220P00150000": -1})

    assert len(drifts) == 1
    assert drifts[0].internal_quantity == -1
    assert drifts[0].broker_quantity == -2


def test_reconcile_positions_no_drift_when_matching(monkeypatch):
    monkeypatch.setattr(
        broker,
        "get_broker_positions",
        lambda: [{"symbol": "AAPL260220P00150000", "quantity": -1}],
    )

    drifts = broker.reconcile_positions({"AAPL260220P00150000": -1})

    assert drifts == []


def test_reconcile_positions_flags_broker_only_symbol(monkeypatch):
    monkeypatch.setattr(
        broker,
        "get_broker_positions",
        lambda: [{"symbol": "UNKNOWN123", "quantity": 1}],
    )

    drifts = broker.reconcile_positions({})

    assert len(drifts) == 1
    assert drifts[0].symbol == "UNKNOWN123"
    assert drifts[0].internal_quantity == 0.0
