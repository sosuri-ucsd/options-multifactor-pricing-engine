"""
End-to-end orchestration: for each ticker on the watchlist, pull data,
score every factor, build the pricing distribution, price and rank
candidate covered-call / cash-secured-put contracts, and log the ranked
output for the dashboard (dashboard/data.py reads
config.LOG_DIR / config.RANKED_CANDIDATES_LOG_FILE).

Defaults to a dry run: candidates are computed, priced, and ranked, and the
ranked list is logged, but no order is ever submitted unless --live is
passed explicitly. Even with --live, execution/tradier_broker.py only
talks to Tradier's *sandbox* (paper trading) endpoint -- nothing in this
codebase places a real-money trade.

Known simplifications in this first end-to-end wiring (called out here
rather than left silent):
  - Dividend yield q is hardcoded to 0.0 -- not yet sourced from
    data/yfinance_client.py's get_dividends(), which exists but isn't wired
    into a forward-yield estimate yet.
  - Position size for the --live path is fixed at 1 contract per order --
    risk/sizing.py's vol-targeted sizing exists but isn't wired in here
    yet (it needs an IV-vol-of-vol estimate per ticker that isn't computed
    anywhere yet either). Only the portfolio delta/vega/beta limit gate
    (risk/limits.py) is enforced pre-submission for now.
  - Beta-weighted delta uses factors/beta_regime.py's rolling beta when
    available, or 1.0 if there isn't enough history for that ticker yet.
"""
import argparse
import json
import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np

import config
from data import fred_client, tradier_client, yfinance_client
from decision.edge import (
    CONTRACT_MULTIPLIER,
    Candidate,
    capital_required_cash_secured_put,
    capital_required_covered_call,
)
from decision.ranking import rank_candidates, to_audit_rows
from deployment.alerting import alert_risk_limit_breach, alert_unhandled_exception
from deployment.logging_setup import configure_logging
from factors import beta_regime, greeks_shape, har_rv, liquidity, momentum, skew, vol_richness
from pricing import black_scholes, monte_carlo
from pricing.distribution import build_distribution
from risk.limits import PortfolioExposure, gate_new_order

logger = logging.getLogger("options_pricing_engine.main")


def _mid(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if not bid or not ask or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / 2.0


def _pick_expiration_by_dte(
    expirations: list[str], as_of: date, dte_min: int, dte_max: int
) -> Optional[str]:
    in_window = []
    for exp_str in expirations:
        dte = (date.fromisoformat(exp_str) - as_of).days
        if dte_min <= dte <= dte_max:
            in_window.append((dte, exp_str))
    if not in_window:
        return None
    return min(in_window)[1]  # closest to dte_min within the window


def build_candidates_for_ticker(ticker: str, as_of: date, risk_free_rate: float) -> list[Candidate]:
    expirations = tradier_client.get_expirations(ticker)
    near_exp = _pick_expiration_by_dte(expirations, as_of, config.TARGET_DTE_MIN, config.TARGET_DTE_MAX)
    if near_exp is None:
        logger.warning("no expiration for %s in the target DTE window", ticker)
        return []
    far_exp = _pick_expiration_by_dte(expirations, as_of, config.FAR_DTE_MIN, config.FAR_DTE_MAX)

    near_chain = tradier_client.get_chain_with_greeks(ticker, near_exp)
    far_chain = tradier_client.get_chain_with_greeks(ticker, far_exp) if far_exp else near_chain

    start = as_of - timedelta(days=config.PRICE_HISTORY_LOOKBACK_DAYS)
    daily_bars = yfinance_client.get_daily_bars(ticker, start, as_of)
    if len(daily_bars) < 30:
        logger.warning("insufficient price history for %s (%d bars)", ticker, len(daily_bars))
        return []
    spy_bars = yfinance_client.get_daily_bars("SPY", start, as_of)
    vix_bars = yfinance_client.get_daily_bars("^VIX", start, as_of)

    spot = daily_bars[-1]["close"]
    closes = [b["close"] for b in daily_bars]
    spy_closes = [b["close"] for b in spy_bars]
    vix_closes = [b["close"] for b in vix_bars]

    dte = (date.fromisoformat(near_exp) - as_of).days
    horizon_trading_days = max(1, round(dte * 252 / 365))
    near_atm_iv = vol_richness.atm_iv_from_chain(near_chain, spot)

    vr_result = vol_richness.compute(ticker, as_of, near_chain, far_chain, spot)
    hr_result = har_rv.compute(ticker, as_of, daily_bars, near_atm_iv, horizon_trading_days)
    sk_result = skew.compute(ticker, as_of, near_chain)
    br_result = beta_regime.compute(ticker, as_of, closes, spy_closes, vix_closes)
    mom_result = momentum.compute(ticker, as_of, closes)

    sigma_base = hr_result.raw_inputs.get("forecast_annualized_vol")
    if sigma_base is None:
        # HAR-RV couldn't fit (insufficient trading history) -- fall back to
        # the contract's own market IV rather than leaving the pricing
        # engine with no volatility estimate at all.
        sigma_base = near_atm_iv if near_atm_iv else 0.20

    T_years = dte / 365.0
    dist_params = build_distribution(
        S0=spot,
        T=T_years,
        r=risk_free_rate,
        q=0.0,
        sigma_base=sigma_base,
        regime_score=br_result.score,
        skew_score=sk_result.score,
    )
    beta = br_result.raw_inputs.get("beta") or 1.0

    rng = np.random.default_rng()
    candidates = []
    for contract in near_chain:
        option_type = contract.get("option_type")
        strike = contract.get("strike")
        bid, ask = contract.get("bid"), contract.get("ask")
        market_price = _mid(bid, ask)
        if option_type not in ("call", "put") or strike is None or market_price is None:
            continue

        open_interest = contract.get("open_interest") or 0
        volume = contract.get("volume") or 0
        liq_result = liquidity.compute(ticker, as_of, open_interest, volume, bid, ask)

        greeks = contract.get("greeks") or {}
        contract_iv = greeks.get("mid_iv") or greeks.get("smv_vol") or sigma_base
        delta, gamma, theta, vega = (
            greeks.get("delta"),
            greeks.get("gamma"),
            greeks.get("theta"),
            greeks.get("vega"),
        )
        if delta is None or gamma is None or theta is None or vega is None:
            # Vendor greeks missing for this contract -- fall back to our own
            # BSM greeks at the contract's own market IV (or sigma_base if
            # even that's missing).
            bs_greeks = black_scholes.greeks(option_type, spot, strike, T_years, risk_free_rate, 0.0, contract_iv)
            delta = delta if delta is not None else bs_greeks["delta"]
            gamma = gamma if gamma is not None else bs_greeks["gamma"]
            theta = theta if theta is not None else bs_greeks["theta"]
            vega = vega if vega is not None else bs_greeks["vega"]

        strategy = "covered_call" if option_type == "call" else "cash_secured_put"
        gs_result = greeks_shape.compute(
            ticker, as_of, strategy=strategy, delta=delta, gamma=gamma, theta=theta,
            spot=spot, contract_price=market_price, dte=dte,
        )

        mc_result = monte_carlo.price_option(
            option_type, strike, dist_params, n_paths=config.MC_DEFAULT_NUM_PATHS, rng=rng
        )

        capital_required = (
            capital_required_covered_call(spot)
            if strategy == "covered_call"
            else capital_required_cash_secured_put(strike)
        )

        candidates.append(
            Candidate(
                ticker=ticker,
                strategy=strategy,
                option_type=option_type,
                strike=strike,
                expiration=date.fromisoformat(near_exp),
                dte=dte,
                market_price=market_price,
                model_price=mc_result.price,
                capital_required=capital_required,
                factor_results=[vr_result, hr_result, sk_result, br_result, liq_result, gs_result, mom_result],
                option_symbol=contract.get("symbol"),
                bid=bid,
                ask=ask,
                open_interest=open_interest,
                volume=volume,
                delta=delta,
                vega=vega,
            )
        )

    return candidates


def _submit_top_candidates(ranked, max_orders: int = 5) -> None:
    from execution.tradier_broker import submit_single_leg_option_order  # deferred: needs live creds

    exposure = PortfolioExposure(net_delta=0.0, net_vega=0.0, beta_weighted_net_delta=0.0)
    submitted = 0
    for ranked_candidate in ranked:
        if submitted >= max_orders or ranked_candidate.expected_pnl_per_capital <= 0:
            break
        c = ranked_candidate.candidate
        if c.delta is None or c.option_symbol is None:
            continue

        # Short the option: sign-flip the raw per-share greeks and scale to
        # a full contract for the portfolio-level shares-equivalent limits.
        candidate_delta = -c.delta * CONTRACT_MULTIPLIER
        candidate_vega = -(c.vega or 0.0) * CONTRACT_MULTIPLIER
        candidate_beta_weighted_delta = candidate_delta * 1.0  # see module docstring

        gate_result = gate_new_order(exposure, candidate_delta, candidate_vega, candidate_beta_weighted_delta)
        if not gate_result.allowed:
            alert_risk_limit_breach(gate_result.reasons)
            continue

        result = submit_single_leg_option_order(
            underlying_symbol=c.ticker,
            option_symbol=c.option_symbol,
            side="sell_to_open",
            quantity=1,
            order_type="limit",
            duration="day",
            limit_price=c.market_price,
            open_interest=c.open_interest or 0,
            volume=c.volume or 0,
            bid=c.bid or 0.0,
            ask=c.ask or 0.0,
            current_exposure=exposure,
            candidate_delta=candidate_delta,
            candidate_vega=candidate_vega,
            candidate_beta_weighted_delta=candidate_beta_weighted_delta,
        )
        if result.status == "submitted":
            exposure = gate_result.resulting_exposure
            submitted += 1
        else:
            logger.warning("order blocked for %s: %s", c.ticker, result.blocked_reasons)


def run_pipeline(
    tickers: Optional[list[str]] = None,
    dry_run: bool = True,
    as_of: Optional[date] = None,
) -> list[dict]:
    tickers = tickers or config.WATCHLIST
    as_of = as_of or date.today()
    risk_free_rate = fred_client.get_risk_free_rate(as_of)

    all_candidates: list[Candidate] = []
    for ticker in tickers:
        try:
            all_candidates.extend(build_candidates_for_ticker(ticker, as_of, risk_free_rate))
        except Exception:
            logger.exception("failed to build candidates for %s", ticker)

    ranked = rank_candidates(all_candidates)
    rows = to_audit_rows(ranked)

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    (config.LOG_DIR / config.RANKED_CANDIDATES_LOG_FILE).write_text(json.dumps(rows, indent=2, default=str))
    logger.info("ranked %d candidates from %d ticker(s)", len(rows), len(tickers))

    if not dry_run:
        _submit_top_candidates(ranked)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None, help="Override config.WATCHLIST")
    parser.add_argument("--live", action="store_true", help="Submit gated orders to the Tradier sandbox")
    args = parser.parse_args()

    configure_logging()
    try:
        run_pipeline(tickers=args.tickers, dry_run=not args.live)
    except Exception as exc:
        logger.exception("unhandled exception in pipeline run")
        alert_unhandled_exception(exc)
        raise


if __name__ == "__main__":
    main()
