import json
from datetime import date, datetime, timedelta

import main
import config


def _bars(n, base=100.0, step=0.05):
    """Simple, slightly wobbling synthetic daily bars -- enough for the
    liquidity/greeks/ranking wiring to exercise every step without needing
    a real HAR-RV fit (which needs ~250+ observations, covered separately
    in tests/factors/test_har_rv.py)."""
    bars = []
    price = base
    for i in range(n):
        wobble = step if i % 2 == 0 else -step
        price = price + wobble
        bars.append({"date": "2026-01-01", "open": price, "high": price + 0.2, "low": price - 0.2, "close": price, "volume": 1_000_000})
    return bars


def _contract(option_type, strike, bid, ask, oi=1000, volume=200, delta=0.30, gamma=0.02, theta=-0.05, vega=0.15, mid_iv=0.25):
    return {
        "symbol": f"TEST260101{'C' if option_type == 'call' else 'P'}00{int(strike)}000",
        "option_type": option_type,
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "open_interest": oi,
        "volume": volume,
        "greeks": {"delta": delta if option_type == "call" else -delta, "gamma": gamma, "theta": theta, "vega": vega, "mid_iv": mid_iv},
    }


def _patch_data_layer(monkeypatch, as_of, near_exp, far_exp):
    monkeypatch.setattr(main.tradier_client, "get_expirations", lambda symbol: [near_exp.isoformat(), far_exp.isoformat()])

    good_call = _contract("call", 105.0, bid=2.00, ask=2.10)
    good_put = _contract("put", 95.0, bid=1.80, ask=1.90)
    illiquid_put = _contract("put", 80.0, bid=0.05, ask=2.00, oi=1, volume=0)
    near_chain = [good_call, good_put, illiquid_put]
    far_chain = [_contract("call", 110.0, bid=3.0, ask=3.2, mid_iv=0.22)]

    def fake_chain(symbol, expiration):
        return near_chain if expiration == near_exp.isoformat() else far_chain

    monkeypatch.setattr(main.tradier_client, "get_chain_with_greeks", fake_chain)

    def fake_bars(ticker, start, end):
        base = {"TEST": 100.0, "SPY": 400.0, "^VIX": 18.0}.get(ticker, 100.0)
        return _bars(60, base=base)

    monkeypatch.setattr(main.yfinance_client, "get_daily_bars", fake_bars)
    monkeypatch.setattr(main.yfinance_client, "get_dividends", lambda ticker: [])
    monkeypatch.setattr(main.fred_client, "get_risk_free_rate", lambda as_of=None: 0.05)


def test_run_pipeline_dry_run_writes_ranked_log_and_excludes_illiquid_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    rows = main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)

    assert len(rows) == 2  # good_call + good_put; illiquid_put gated out
    strikes = {row["strike"] for row in rows}
    assert strikes == {105.0, 95.0}

    logged = json.loads((tmp_path / config.RANKED_CANDIDATES_LOG_FILE).read_text())
    assert logged == rows
    for row in rows:
        assert "contributing_factors" in row
        assert "liquidity" in row["contributing_factors"]


def test_run_pipeline_dry_run_never_touches_broker(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    calls = []
    monkeypatch.setattr(
        "execution.tradier_broker.submit_single_leg_option_order",
        lambda **kwargs: calls.append(kwargs),
    )

    main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)

    assert calls == []


def test_run_pipeline_live_submits_gated_orders(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    from execution.tradier_broker import OrderResult

    calls = []

    def fake_submit(**kwargs):
        calls.append(kwargs)
        return OrderResult(order_id="1", status="submitted", blocked_reasons=[])

    monkeypatch.setattr("execution.tradier_broker.submit_single_leg_option_order", fake_submit)

    main.run_pipeline(tickers=["TEST"], dry_run=False, as_of=as_of)

    assert len(calls) >= 1
    assert all(c["side"] == "sell_to_open" for c in calls)


def test_run_pipeline_handles_ticker_with_no_expiration_in_window(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    monkeypatch.setattr(main.tradier_client, "get_expirations", lambda symbol: ["2026-06-05"])  # too soon, outside window
    monkeypatch.setattr(main.fred_client, "get_risk_free_rate", lambda as_of=None: 0.05)

    rows = main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)

    assert rows == []


def test_run_pipeline_appends_to_history_log(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)
    main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)

    lines = (tmp_path / config.CANDIDATES_HISTORY_LOG_FILE).read_text().strip().splitlines()
    assert len(lines) == 2  # appended, not overwritten
    for line in lines:
        entry = json.loads(line)
        assert "timestamp" in entry
        assert len(entry["candidates"]) == 2


def test_beta_fallback_is_logged_when_history_insufficient(tmp_path, monkeypatch, caplog):
    # The synthetic bars helper only produces 60 days -- well short of
    # BETA_ROLLING_WINDOW_DAYS (126), so beta_regime can't compute a real
    # rolling beta and main.py must fall back to 1.0, loudly.
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    with caplog.at_level("WARNING"):
        main.run_pipeline(tickers=["TEST"], dry_run=True, as_of=as_of)

    assert any("beta fallback to 1.0" in r.message for r in caplog.records)


def test_live_order_defaults_to_one_contract_with_fresh_iv_history(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    from execution.tradier_broker import OrderResult

    calls = []
    monkeypatch.setattr(
        "execution.tradier_broker.submit_single_leg_option_order",
        lambda **kwargs: (calls.append(kwargs), OrderResult(order_id="1", status="submitted", blocked_reasons=[]))[1],
    )

    with caplog.at_level("WARNING"):
        main.run_pipeline(tickers=["TEST"], dry_run=False, as_of=as_of)

    assert len(calls) >= 1
    assert all(c["quantity"] == 1 for c in calls)
    assert any("insufficient IV history for vol-targeted sizing" in r.message for r in caplog.records)


def test_live_order_uses_vol_targeted_sizing_with_seeded_iv_history(tmp_path, monkeypatch, caplog):
    from factors import vol_richness

    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    as_of = date(2026, 6, 1)
    near_exp = as_of + timedelta(days=30)
    far_exp = as_of + timedelta(days=75)
    _patch_data_layer(monkeypatch, as_of, near_exp, far_exp)

    # Seed 20 days of bouncing IV history for TEST so iv_vol_of_vol has enough
    # to compute a real (nonzero) estimate instead of falling back.
    for i in range(20, 0, -1):
        wobble = 0.01 if i % 2 == 0 else -0.01
        vol_richness.record_daily_atm_iv("TEST", as_of - timedelta(days=i), 0.25 + wobble)

    from execution.tradier_broker import OrderResult

    calls = []
    monkeypatch.setattr(
        "execution.tradier_broker.submit_single_leg_option_order",
        lambda **kwargs: (calls.append(kwargs), OrderResult(order_id="1", status="submitted", blocked_reasons=[]))[1],
    )

    with caplog.at_level("WARNING"):
        main.run_pipeline(tickers=["TEST"], dry_run=False, as_of=as_of, account_capital=1_000_000.0)

    assert len(calls) >= 1
    assert not any("insufficient IV history for vol-targeted sizing" in r.message for r in caplog.records)


def test_is_market_hours():
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(config.MARKET_TIMEZONE)
    weekday_open = datetime(2026, 6, 1, 10, 0, tzinfo=tz)  # Monday, 10am ET
    weekday_before_open = datetime(2026, 6, 1, 8, 0, tzinfo=tz)
    weekend = datetime(2026, 6, 6, 10, 0, tzinfo=tz)  # Saturday

    assert main._is_market_hours(weekday_open) is True
    assert main._is_market_hours(weekday_before_open) is False
    assert main._is_market_hours(weekend) is False


def test_run_scheduled_loop_skips_outside_market_hours(monkeypatch):
    from zoneinfo import ZoneInfo

    calls = []
    monkeypatch.setattr(main, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        main,
        "datetime",
        type("FakeDatetime", (), {"now": staticmethod(lambda tz=None: datetime(2026, 6, 6, 10, 0, tzinfo=tz))}),
    )

    main.run_scheduled_loop(tickers=["TEST"], max_iterations=2)

    assert calls == []  # Saturday -- never runs


def test_run_scheduled_loop_runs_during_market_hours(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "run_pipeline", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        main,
        "datetime",
        type("FakeDatetime", (), {"now": staticmethod(lambda tz=None: datetime(2026, 6, 1, 10, 0, tzinfo=tz))}),
    )

    main.run_scheduled_loop(tickers=["TEST"], max_iterations=3)

    assert len(calls) == 3
