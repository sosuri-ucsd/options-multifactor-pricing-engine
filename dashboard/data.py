"""
Read-only data loading for the dashboard. Every function here only reads
JSON log files that the pipeline (main.py, via decision/risk/execution)
writes as it runs -- nothing in this module calls a live API or a broker,
and nothing in the dashboard writes back into the trading system. This
module is kept separate from dashboard/app.py (the Streamlit rendering
layer) specifically so the loading/aggregation logic is unit-testable
without a Streamlit runtime.

Expected log files, written by the pipeline under config.LOG_DIR:
  - ranked_candidates.json: decision.ranking.to_audit_rows(...) output
  - positions.json: one dict per open position, with at least
    {"symbol", "delta", "vega", "beta_weighted_delta", ...}
  - pnl_history.json: [{"date": "YYYY-MM-DD", "live_pnl": float,
    "backtested_expected_pnl": float}, ...]

Every loader returns an empty list if its log file doesn't exist yet
(e.g. before the pipeline has ever run) rather than raising, so the
dashboard can render a friendly empty state.
"""
import json
from pathlib import Path

import config


def _load_json(filename: str) -> list[dict]:
    path = config.LOG_DIR / filename
    if not path.exists():
        return []
    return json.loads(path.read_text())


def load_ranked_candidates() -> list[dict]:
    return _load_json("ranked_candidates.json")


def load_positions() -> list[dict]:
    return _load_json("positions.json")


def load_pnl_history() -> list[dict]:
    return _load_json("pnl_history.json")


def portfolio_exposure_summary(positions: list[dict]) -> dict:
    return {
        "net_delta": sum(p.get("delta", 0.0) for p in positions),
        "net_vega": sum(p.get("vega", 0.0) for p in positions),
        "beta_weighted_net_delta": sum(p.get("beta_weighted_delta", 0.0) for p in positions),
    }
