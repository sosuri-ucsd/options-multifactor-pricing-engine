"""
Streamlit dashboard: read-only over the pipeline's own logs (dashboard/data.py).
No write path back into the trading system -- this process never calls a
broker or a data-vendor API, and never places or cancels an order.

Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# `streamlit run dashboard/app.py` puts this file's own directory
# (dashboard/) at the front of sys.path, not the project root -- so
# `dashboard` (the package containing this very file) can't be resolved
# without adding the parent directory explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from dashboard.data import load_pnl_history, load_positions, load_ranked_candidates, portfolio_exposure_summary


def render() -> None:
    st.set_page_config(page_title="Options Multi-Factor Pricing Engine", layout="wide")
    st.title("Options Multi-Factor Pricing & Decision Engine")
    st.caption("Read-only view over pipeline logs. This page cannot place, modify, or cancel orders.")

    st.header("Ranked candidate trades")
    candidates = load_ranked_candidates()
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True)
    else:
        st.info("No ranked candidates logged yet -- run the pipeline (main.py) to populate this.")

    st.header("Live open positions")
    positions = load_positions()
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)
        exposure = portfolio_exposure_summary(positions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Net delta", f"{exposure['net_delta']:.1f}")
        col2.metric("Net vega", f"{exposure['net_vega']:.1f}")
        col3.metric("Beta-weighted net delta", f"{exposure['beta_weighted_net_delta']:.1f}")
    else:
        st.info("No open positions logged yet.")

    st.header("P&L: live paper trading vs. backtested expectation")
    pnl_history = load_pnl_history()
    if pnl_history:
        df = pd.DataFrame(pnl_history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        st.line_chart(df[["live_pnl", "backtested_expected_pnl"]])
    else:
        st.info("No P&L history logged yet.")


if __name__ == "__main__":
    render()
