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

Remaining known simplification: beta-weighted delta uses
factors/beta_regime.py's rolling beta when available, or falls back to 1.0
when there isn't enough history for that ticker yet -- every fallback is
logged as a warning (see _resolve_beta) rather than happening silently, so
a decision made on incomplete data is always visible in the logs.
"""
import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

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
from risk import sizing
from risk.limits import PortfolioExposure, gate_new_order

logger = logging.getLogger("options_pricing_engine.main")


def _resolve_beta(ticker: str, br_result) -> float:
    beta = br_result.raw_inputs.get("beta")
    if beta is None:
        logger.warning(
            "beta fallback to 1.0 for %s -- insufficient rolling-beta history; "
            "beta-weighted delta for this ticker is not based on a real estimate",
            ticker,
        )
        return 1.0
    return beta


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

    dividend_yield = yfinance_client.estimate_dividend_yield(ticker, as_of, spot)

    T_years = dte / 365.0
    dist_params = build_distribution(
        S0=spot,
        T=T_years,
        r=risk_free_rate,
        q=dividend_yield,
        sigma_base=sigma_base,
        regime_score=br_result.score,
        skew_score=sk_result.score,
    )
    beta = _resolve_beta(ticker, br_result)

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
            bs_greeks = black_scholes.greeks(option_type, spot, strike, T_years, risk_free_rate, dividend_yield, contract_iv)
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
                beta=beta,
            )
        )

    return candidates


def _resolve_quantity(c: Candidate, as_of: date, account_capital: float, portfolio_vol_used: float) -> int:
    """Vol-targeted contract count for this candidate (risk/sizing.py), given
    how much of the portfolio vol budget prior orders in this run already
    used. Falls back to a conservative 1 contract -- logged, not silent --
    when there isn't enough IV history for this ticker to estimate vol-of-vol
    yet (a new/rarely-scored ticker, or the very first run before any IV
    history has accumulated)."""
    if c.vega is None:
        return 1
    iv_history = vol_richness.load_iv_history(c.ticker, as_of)
    vol_of_vol = vol_richness.iv_vol_of_vol(iv_history)
    if vol_of_vol is None:
        logger.warning(
            "insufficient IV history for vol-targeted sizing on %s -- defaulting to 1 contract",
            c.ticker,
        )
        return 1
    vega_per_contract = c.vega * CONTRACT_MULTIPLIER
    remaining_budget = config.PORTFOLIO_VOL_BUDGET_ANNUAL * account_capital - portfolio_vol_used
    max_contracts = sizing.max_contracts_within_vol_budget(vega_per_contract, vol_of_vol, remaining_budget)
    return max(0, min(max_contracts, 10))  # 10-contract cap regardless of budget headroom, a sane starting ceiling


def _submit_top_candidates(
    ranked, as_of: date, account_capital: float, max_orders: int = 5
) -> None:
    from execution.tradier_broker import submit_single_leg_option_order  # deferred: needs live creds

    exposure = PortfolioExposure(net_delta=0.0, net_vega=0.0, beta_weighted_net_delta=0.0)
    portfolio_vol_used = 0.0
    submitted = 0
    for ranked_candidate in ranked:
        if submitted >= max_orders or ranked_candidate.expected_pnl_per_capital <= 0:
            break
        c = ranked_candidate.candidate
        if c.delta is None or c.option_symbol is None:
            continue

        quantity = _resolve_quantity(c, as_of, account_capital, portfolio_vol_used)
        if quantity <= 0:
            logger.info("skipping %s -- vol budget exhausted for this run", c.ticker)
            continue

        # Short the option: sign-flip the raw per-share greeks and scale to
        # a full position for the portfolio-level shares-equivalent limits.
        beta = c.beta if c.beta is not None else 1.0
        candidate_delta = -c.delta * CONTRACT_MULTIPLIER * quantity
        candidate_vega = -(c.vega or 0.0) * CONTRACT_MULTIPLIER * quantity
        candidate_beta_weighted_delta = candidate_delta * beta

        gate_result = gate_new_order(exposure, candidate_delta, candidate_vega, candidate_beta_weighted_delta)
        if not gate_result.allowed:
            alert_risk_limit_breach(gate_result.reasons)
            continue

        result = submit_single_leg_option_order(
            underlying_symbol=c.ticker,
            option_symbol=c.option_symbol,
            side="sell_to_open",
            quantity=quantity,
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
            if c.vega is not None:
                vol_of_vol = vol_richness.iv_vol_of_vol(vol_richness.load_iv_history(c.ticker, as_of))
                if vol_of_vol is not None:
                    portfolio_vol_used += sizing.contract_vega_dollar_vol(
                        c.vega * CONTRACT_MULTIPLIER, vol_of_vol
                    ) * quantity
            submitted += 1
        else:
            logger.warning("order blocked for %s: %s", c.ticker, result.blocked_reasons)


def run_pipeline(
    tickers: Optional[list[str]] = None,
    dry_run: bool = True,
    as_of: Optional[date] = None,
    account_capital: Optional[float] = None,
) -> list[dict]:
    tickers = tickers or config.WATCHLIST
    as_of = as_of or date.today()
    account_capital = account_capital if account_capital is not None else config.DEFAULT_ACCOUNT_CAPITAL
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
    with open(config.LOG_DIR / config.CANDIDATES_HISTORY_LOG_FILE, "a") as history_file:
        history_file.write(json.dumps({"timestamp": datetime.now().isoformat(), "candidates": rows}, default=str) + "\n")
    logger.info("ranked %d candidates from %d ticker(s)", len(rows), len(tickers))

    if not dry_run:
        _submit_top_candidates(ranked, as_of=as_of, account_capital=account_capital)

    return rows


def _is_market_hours(now: datetime) -> bool:
    if now.weekday() >= 5:
        return False
    open_time = now.replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    close_time = now.replace(hour=config.MARKET_CLOSE_HOUR, minute=config.MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    return open_time <= now <= close_time


def run_scheduled_loop(
    tickers: Optional[list[str]] = None,
    dry_run: bool = True,
    account_capital: Optional[float] = None,
    interval_minutes: int = config.DEFAULT_LOOP_INTERVAL_MINUTES,
    max_iterations: Optional[int] = None,
) -> None:
    """Re-runs run_pipeline every interval_minutes, skipping iterations
    outside market hours rather than running (and re-logging stale-market
    candidates) around the clock. Each run appends to
    config.CANDIDATES_HISTORY_LOG_FILE rather than only overwriting the
    latest snapshot, so a visitor loading a downstream view later in the
    day sees an updated history, not a single static demo run.
    max_iterations is for tests/manual bounded runs; leave None to loop
    indefinitely (the intended real usage under a scheduler/systemd unit).
    """
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        now = datetime.now(ZoneInfo(config.MARKET_TIMEZONE))
        if _is_market_hours(now):
            try:
                run_pipeline(tickers=tickers, dry_run=dry_run, account_capital=account_capital)
            except Exception:
                logger.exception("pipeline iteration failed")
        else:
            logger.info("outside market hours (%s) -- skipping this iteration", now.strftime("%Y-%m-%d %H:%M %Z"))
        iteration += 1
        if max_iterations is None or iteration < max_iterations:
            time.sleep(interval_minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=None, help="Override config.WATCHLIST")
    parser.add_argument("--live", action="store_true", help="Submit gated orders to the Tradier sandbox")
    parser.add_argument("--loop", action="store_true", help="Run repeatedly on --interval-minutes during market hours instead of once")
    parser.add_argument("--interval-minutes", type=int, default=config.DEFAULT_LOOP_INTERVAL_MINUTES)
    parser.add_argument("--account-capital", type=float, default=None)
    args = parser.parse_args()

    configure_logging()
    try:
        if args.loop:
            run_scheduled_loop(
                tickers=args.tickers,
                dry_run=not args.live,
                account_capital=args.account_capital,
                interval_minutes=args.interval_minutes,
            )
        else:
            run_pipeline(tickers=args.tickers, dry_run=not args.live, account_capital=args.account_capital)
    except Exception as exc:
        logger.exception("unhandled exception in pipeline run")
        alert_unhandled_exception(exc)
        raise


if __name__ == "__main__":
    main()
