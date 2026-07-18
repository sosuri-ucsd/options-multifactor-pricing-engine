"""
IV-vs-forecast factor: HAR-RV (Heterogeneous Autoregressive Realized
Volatility, Corsi 2009).

Step 1 -- daily realized variance from OHLC.
We only have daily bars (yfinance), not intraday ticks, so a single
close-to-close squared return is a noisy realized-variance proxy. The
Garman-Klass estimator uses the full daily range plus open/close and is
substantially more efficient (lower variance for the same data) than
close-to-close when only daily OHLC is available:

    sigma^2_GK = 0.5 * ln(H/L)^2 - (2*ln2 - 1) * ln(C/O)^2

This is today's realized variance, RV_t, on the scale of daily log-return
variance.

Step 2 -- HAR-RV regression.
Corsi's empirical finding is that realized volatility displays long-memory
persistence (today's vol is correlated with vol from weeks and months ago,
decaying slowly), which a full long-memory model (ARFIMA) would normally be
needed to capture. HAR-RV approximates that long memory cheaply with a
plain linear regression on RV averaged over three horizons -- daily,
weekly, monthly -- as of day t:

    RV_D_t = RV_t                          (today)
    RV_W_t = mean(RV_{t-4}, ..., RV_t)      (trailing 5 trading days)
    RV_M_t = mean(RV_{t-21}, ..., RV_t)     (trailing 22 trading days)

and regresses a future realized-variance target on those three features:

    y_t = b0 + bD * RV_D_t + bW * RV_W_t + bM * RV_M_t + eps_t

The target y_t is the average realized variance over the next `horizon`
trading days (not a one-step-ahead forecast iterated forward) because what
this factor actually needs is a forecast of realized vol over the specific
option's remaining life -- e.g. if an expiration is 20 trading days out,
fit and predict a 20-day-forward-average target directly, rather than
compounding one-day-ahead forecasts and accumulating their error.

Fit is plain OLS (via numpy.linalg.lstsq) -- three regressors and an
intercept does not need anything heavier.

Step 3 -- score.
Forecast realized vol is annualized (forecast_daily_variance * 252, then
sqrt) and compared against the current market ATM IV for the matching
expiration. If market IV sits well above what the model expects to
realize, the market is pricing in more vol than HAR-RV thinks will show up
-- favorable for selling premium (positive score); market IV below the
forecast is unfavorable (negative score, the model expects more realized
vol than is being paid for).
"""
import math
from datetime import date as date_type
from typing import Optional

import numpy as np

import config
from factors.base import FactorResult, clip_score

_TRADING_DAYS_PER_YEAR = 252
# A 20% relative gap between market IV and forecast vol maps to a full +-1 score.
_SCORE_SCALE = 0.20


def garman_klass_daily_variance(bars: list[dict]) -> list[float]:
    """bars: chronological daily OHLC dicts (data.yfinance_client.get_daily_bars
    output). Returns one Garman-Klass variance estimate per bar."""
    variances = []
    for bar in bars:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            variances.append(0.0)
            continue
        log_hl = math.log(h / l)
        log_co = math.log(c / o)
        variance = 0.5 * log_hl**2 - (2 * math.log(2) - 1) * log_co**2
        variances.append(max(variance, 0.0))
    return variances


def build_har_features(
    daily_variances: list[float],
    weekly_lag_days: int = config.HAR_RV_WEEKLY_LAG_DAYS,
    monthly_lag_days: int = config.HAR_RV_MONTHLY_LAG_DAYS,
) -> list[tuple[float, float, float]]:
    """One (RV_D, RV_W, RV_M) feature row per day t, for every t with enough
    trailing history to compute the monthly average (t >= monthly_lag_days - 1)."""
    features = []
    for t in range(monthly_lag_days - 1, len(daily_variances)):
        rv_d = daily_variances[t]
        rv_w = sum(daily_variances[t - weekly_lag_days + 1 : t + 1]) / weekly_lag_days
        rv_m = sum(daily_variances[t - monthly_lag_days + 1 : t + 1]) / monthly_lag_days
        features.append((rv_d, rv_w, rv_m))
    return features


def fit_har_rv(
    daily_variances: list[float],
    horizon_days: int,
    monthly_lag_days: int = config.HAR_RV_MONTHLY_LAG_DAYS,
    weekly_lag_days: int = config.HAR_RV_WEEKLY_LAG_DAYS,
) -> Optional[np.ndarray]:
    """OLS-fit [b0, bD, bW, bM] regressing the average realized variance over
    the next `horizon_days` on today's (RV_D, RV_W, RV_M). Returns None if
    there isn't enough history to form config.HAR_RV_MIN_TRAINING_OBS rows
    with both full lookback and full forward horizon available."""
    features = build_har_features(daily_variances, weekly_lag_days, monthly_lag_days)
    # features[i] corresponds to day index (monthly_lag_days - 1 + i) in the
    # original series; its target is the forward average starting the next day.
    rows_x, rows_y = [], []
    for i, (rv_d, rv_w, rv_m) in enumerate(features):
        t = monthly_lag_days - 1 + i
        forward_start, forward_end = t + 1, t + 1 + horizon_days
        if forward_end > len(daily_variances):
            break
        target = sum(daily_variances[forward_start:forward_end]) / horizon_days
        rows_x.append([1.0, rv_d, rv_w, rv_m])
        rows_y.append(target)

    if len(rows_y) < config.HAR_RV_MIN_TRAINING_OBS:
        return None

    x = np.array(rows_x)
    y = np.array(rows_y)
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(x, y, rcond=None)
    return coeffs


def forecast_har_rv(coeffs: np.ndarray, rv_d: float, rv_w: float, rv_m: float) -> float:
    """Predicted average daily realized variance over the fitted horizon.
    Clipped at 0 -- OLS has no floor and can predict a negative variance
    when current RV inputs are unusually low relative to the training fit."""
    forecast = coeffs[0] + coeffs[1] * rv_d + coeffs[2] * rv_w + coeffs[3] * rv_m
    return max(forecast, 0.0)


def compute(
    ticker: str,
    as_of: date_type,
    daily_bars: list[dict],
    current_market_iv: Optional[float],
    horizon_days: int,
) -> FactorResult:
    """daily_bars: chronological OHLC up to and including as_of
    (data.yfinance_client.get_daily_bars output). horizon_days: trading days
    until the option expiration being evaluated -- the forecast horizon
    should match how far out the contract is."""
    variances = garman_klass_daily_variance(daily_bars)
    coeffs = fit_har_rv(variances, horizon_days)

    raw_inputs: dict = {
        "current_market_iv": current_market_iv,
        "horizon_days": horizon_days,
        "n_daily_bars": len(daily_bars),
    }

    if coeffs is None or len(variances) < config.HAR_RV_MONTHLY_LAG_DAYS:
        raw_inputs["reason"] = "insufficient history for HAR-RV fit"
        return FactorResult(
            factor_name="har_rv", ticker=ticker, as_of=as_of, score=0.0, raw_inputs=raw_inputs
        )

    rv_d = variances[-1]
    rv_w = sum(variances[-config.HAR_RV_WEEKLY_LAG_DAYS :]) / config.HAR_RV_WEEKLY_LAG_DAYS
    rv_m = sum(variances[-config.HAR_RV_MONTHLY_LAG_DAYS :]) / config.HAR_RV_MONTHLY_LAG_DAYS

    forecast_daily_variance = forecast_har_rv(coeffs, rv_d, rv_w, rv_m)
    forecast_annualized_vol = math.sqrt(forecast_daily_variance * _TRADING_DAYS_PER_YEAR)

    raw_inputs.update(
        {
            "har_coefficients": {
                "intercept": float(coeffs[0]),
                "daily": float(coeffs[1]),
                "weekly": float(coeffs[2]),
                "monthly": float(coeffs[3]),
            },
            "rv_daily": rv_d,
            "rv_weekly": rv_w,
            "rv_monthly": rv_m,
            "forecast_annualized_vol": forecast_annualized_vol,
        }
    )

    if current_market_iv is None or forecast_annualized_vol == 0:
        score = 0.0
    else:
        relative_gap = (current_market_iv - forecast_annualized_vol) / forecast_annualized_vol
        score = clip_score(relative_gap / _SCORE_SCALE)

    return FactorResult(
        factor_name="har_rv", ticker=ticker, as_of=as_of, score=score, raw_inputs=raw_inputs
    )
