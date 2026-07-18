"""
Central configuration: tunables, thresholds, and API base URLs.

No secrets live here. Credentials are read from environment variables
(populated from .env via python-dotenv) by the data/execution clients
that need them -- this module never touches os.environ for secret values.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DB_PATH = CACHE_DIR / "market_data.sqlite"
LOG_DIR = ROOT_DIR / "logs"

# --- API endpoints (no keys here, just base URLs) -----------------------

POLYGON_BASE_URL = "https://api.polygon.io"
TRADIER_SANDBOX_BASE_URL = "https://sandbox.tradier.com/v1"
TRADIER_PROD_MARKET_DATA_BASE_URL = "https://api.tradier.com/v1"
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# --- Risk-free rate -------------------------------------------------------

FRED_RISK_FREE_SERIES = "DGS3MO"  # 3-month Treasury yield, percent
RISK_FREE_CACHE_TTL_SECONDS = 6 * 60 * 60  # refresh at most every 6h intraday

# --- Liquidity gate (hard filter, factors/liquidity.py) ------------------
# Contracts failing ANY of these are excluded before reaching the pricing
# engine. These are starting points, not calibrated -- tighten once real
# fill data exists.

LIQUIDITY_MIN_OPEN_INTEREST = 100
LIQUIDITY_MIN_VOLUME = 10
LIQUIDITY_MAX_SPREAD_PCT_OF_MID = 0.10  # (ask - bid) / mid must be <= 10%

# --- Vol richness factor -------------------------------------------------

VOL_RICHNESS_LOOKBACK_DAYS = 252  # ~1 trading year for IV rank/percentile

# --- HAR-RV factor --------------------------------------------------------

HAR_RV_DAILY_LAG = 1
HAR_RV_WEEKLY_LAG_DAYS = 5
HAR_RV_MONTHLY_LAG_DAYS = 22
HAR_RV_MIN_TRAINING_OBS = 250  # ~1 year of daily obs before trusting the fit

# --- Skew factor -----------------------------------------------------------

SKEW_DELTA_TARGET = 0.25  # 25-delta risk reversal
SKEW_HISTORY_LOOKBACK_DAYS = 252
SKEW_MIN_HISTORY_OBS = 20  # below this, score neutral rather than a noisy z-score

# --- Greeks/risk-shape factor -----------------------------------------------

GREEKS_SHAPE_TARGET_DELTA = 0.30          # center of the target delta band (magnitude)
GREEKS_SHAPE_DELTA_TOLERANCE = 0.15       # +-band width before the delta-fit score bottoms out
GREEKS_SHAPE_PIN_RISK_DTE_DECAY = 5.0     # days; gamma-driven pin risk weight decays with this scale
GREEKS_SHAPE_PIN_RISK_RELATIVE_SCALE = 0.05  # relative pin risk mapping to a full -1 score
GREEKS_SHAPE_THETA_DAILY_SCALE = 0.02     # daily theta decay as fraction of premium mapping to +1

# --- Beta / regime factor --------------------------------------------------

BETA_ROLLING_WINDOW_DAYS = 126  # ~6 months
VIX_HIGH_REGIME_THRESHOLD = 25.0
VIX_TREND_LOOKBACK_DAYS = 10
VIX_CRISIS_THRESHOLD = 35.0        # above this AND rising -> hard-gate new premium-selling entries
VIX_LEVEL_BASELINE = 20.0          # "calm" reference VIX level for the level component
VIX_LEVEL_SCALE = 20.0             # VIX points above baseline mapping to a full -1 level component
VIX_TREND_SCALE = 0.15             # 15% VIX move over the trend lookback maps to a full +-1
BETA_COMPONENT_SCALE = 1.0         # beta distance from 1.0 mapping to a full +-1 component

# --- Momentum/technical factor -----------------------------------------------

MOMENTUM_LOOKBACK_DAYS = 20      # ~1 month of trading days
MOMENTUM_ZSCORE_SCALE = 2.0      # a 2-sigma move over the lookback maps the score to 0

# --- Distribution construction (pricing/distribution.py) ---------------------
# Two-component lognormal mixture: "normal" regime + "stress" regime.
# regime_score and skew_score below are factors/beta_regime.py and
# factors/skew.py scores (each in [-1, 1]); only their unfavorable/rich
# direction (negative regime_score, positive skew_score) moves these away
# from their base values -- a favorable reading doesn't get extra credit
# beyond the floor, since "calmer than usual" shouldn't itself widen edge.

DIST_STRESS_WEIGHT_BASE = 0.05           # stress-component probability mass in a neutral regime
DIST_STRESS_WEIGHT_SENSITIVITY = 0.15    # extra weight at regime_score == -1 (full crisis reading)
DIST_STRESS_WEIGHT_MIN = 0.02
DIST_STRESS_WEIGHT_MAX = 0.35

DIST_STRESS_VOL_MULTIPLIER_BASE = 1.8         # stress-component sigma = sigma_base * this, neutral regime
DIST_STRESS_VOL_MULTIPLIER_SENSITIVITY = 1.2  # extra multiplier at regime_score == -1

DIST_SKEW_SHIFT_SCALE = 0.15  # skew_score == 1 (max rich put-skew) -> this much extra left-shift
                              # (log-return units, scaled by sqrt(T)) on the stress component's mean

# --- Monte Carlo pricer -----------------------------------------------------

MC_DEFAULT_NUM_PATHS = 100_000
MC_CONVERGENCE_CHECK_PATH_COUNTS = (1_000, 10_000, 50_000, 100_000, 500_000)
MC_RANDOM_SEED = None  # set to an int for reproducible backtests

# --- Risk overlay -----------------------------------------------------------

PORTFOLIO_VOL_BUDGET_ANNUAL = 0.15  # target annualized portfolio vol from options vega
MAX_PORTFOLIO_NET_DELTA = 500          # shares-equivalent
MAX_PORTFOLIO_NET_VEGA = 2_000         # $ per 1 vol point
MAX_PORTFOLIO_BETA_WEIGHTED_DELTA = 750  # SPY-shares-equivalent
DELTA_HEDGE_TRIGGER_THRESHOLD = 100     # shares-equivalent drift before auto-hedge fires

# --- Backtesting costs -------------------------------------------------------

BACKTEST_COMMISSION_PER_CONTRACT = 0.65  # typical retail options commission
BACKTEST_SPREAD_CROSSING_FRACTION = 0.5  # assume fills at mid + 50% of half-spread
BACKTEST_ASSIGNMENT_FEE = 0.0             # most retail brokers (incl. Tradier) don't charge this
BACKTEST_TRADING_DAYS_PER_YEAR = 252

# --- main.py orchestration ---------------------------------------------------

WATCHLIST = ["AAPL", "MSFT", "SPY"]        # placeholder universe; expand once live-tested
TARGET_DTE_MIN = 20                        # "near" expiration window used for candidate contracts
TARGET_DTE_MAX = 45
FAR_DTE_MIN = 60                           # "far" expiration used only for vol_richness term structure
FAR_DTE_MAX = 90
PRICE_HISTORY_LOOKBACK_DAYS = 450          # > HAR_RV_MIN_TRAINING_OBS + monthly lag + horizon, with margin
DEFAULT_ACCOUNT_CAPITAL = 100_000.0        # placeholder; wire to the broker's real balance before going live
RANKED_CANDIDATES_LOG_FILE = "ranked_candidates.json"
CANDIDATES_HISTORY_LOG_FILE = "candidates_history.jsonl"  # append-only, one line per pipeline run
DEFAULT_LOOP_INTERVAL_MINUTES = 10
MARKET_TIMEZONE = "America/New_York"
MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE = 9, 30
MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE = 16, 0

# --- Environment / credentials (names only, values read by clients) ---------

ENV_POLYGON_API_KEY = "POLYGON_API_KEY"
ENV_TRADIER_API_KEY = "TRADIER_API_KEY"
ENV_TRADIER_ACCOUNT_ID = "TRADIER_ACCOUNT_ID"
ENV_FRED_API_KEY = "FRED_API_KEY"


def require_env(var_name: str) -> str:
    """Fetch a required credential from the environment, failing loudly if unset."""
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable {var_name!r}. "
            f"Copy .env.example to .env and fill in real values."
        )
    return value
