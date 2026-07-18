import json

import config
from dashboard.data import (
    load_pnl_history,
    load_positions,
    load_ranked_candidates,
    portfolio_exposure_summary,
)


def _write_log(tmp_path, monkeypatch, filename, content):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    (tmp_path / filename).write_text(json.dumps(content))


def test_load_ranked_candidates_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    assert load_ranked_candidates() == []


def test_load_ranked_candidates_reads_logged_file(tmp_path, monkeypatch):
    rows = [{"ticker": "AAPL", "expected_pnl_per_capital": 0.02}]
    _write_log(tmp_path, monkeypatch, "ranked_candidates.json", rows)
    assert load_ranked_candidates() == rows


def test_load_positions_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    assert load_positions() == []


def test_load_positions_reads_logged_file(tmp_path, monkeypatch):
    rows = [{"symbol": "AAPL260220P00150000", "delta": -0.3}]
    _write_log(tmp_path, monkeypatch, "positions.json", rows)
    assert load_positions() == rows


def test_load_pnl_history_reads_logged_file(tmp_path, monkeypatch):
    rows = [{"date": "2026-06-01", "live_pnl": 120.0, "backtested_expected_pnl": 100.0}]
    _write_log(tmp_path, monkeypatch, "pnl_history.json", rows)
    assert load_pnl_history() == rows


def test_portfolio_exposure_summary_sums_across_positions():
    positions = [
        {"delta": -30, "vega": 40, "beta_weighted_delta": -25},
        {"delta": -20, "vega": 10, "beta_weighted_delta": -15},
    ]
    summary = portfolio_exposure_summary(positions)
    assert summary == {"net_delta": -50, "net_vega": 50, "beta_weighted_net_delta": -40}


def test_portfolio_exposure_summary_empty_positions():
    assert portfolio_exposure_summary([]) == {
        "net_delta": 0.0,
        "net_vega": 0.0,
        "beta_weighted_net_delta": 0.0,
    }


def test_portfolio_exposure_summary_missing_fields_default_to_zero():
    summary = portfolio_exposure_summary([{"symbol": "AAPL"}])
    assert summary["net_delta"] == 0.0
