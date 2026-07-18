# Options Multi-Factor Pricing & Decision Engine

Personal research project. Generates its own theoretical fair-value price for
equity options — instead of trusting the market's implied volatility — and
trades the gap between the model's price and the market's quoted price.

This is a research project intended to eventually trade small real capital.
It is not investment advice, and nothing here should be relied on by anyone
other than the author.

## How it works

Five layers:

1. **Factors** (`factors/`) — independent, auditable reads on current market
   conditions: vol richness, realized-vol forecast (HAR-RV), skew, Greeks/risk
   shape, beta/regime, liquidity, momentum.
2. **Distribution** (`pricing/distribution.py`) — factor outputs are combined
   into a custom forecast distribution for the underlying at expiration,
   rather than assuming market-implied vol is correct.
3. **Pricing** (`pricing/`) — every candidate contract is priced under the
   custom distribution via Monte Carlo, with a Black-Scholes-Merton run
   alongside as a cross-check against vanilla market-implied pricing.
4. **Decision** (`decision/`) — model price minus market price is the edge;
   candidates are ranked by expected P&L per unit of capital, after a hard
   liquidity/beta-regime gate.
5. **Execution & risk** (`risk/`, `execution/`) — vol-targeted position
   sizing, portfolio Greek/beta limits enforced before order submission,
   broker paper trading via Tradier, with a read-only Streamlit dashboard
   (`dashboard/`) on top.

Initial strategies are single-leg and broker-approval-friendly: covered calls
and cash-secured puts. The factor/pricing engine decides underlying, strike,
and expiration — there is no fixed weekly mechanic.

## Build order / status

- [x] Phase 1 — Data layer (`data/`): Polygon, Tradier, yfinance, FRED, local cache
- [x] Phase 2 — Factors (`factors/`), one at a time
- [x] Phase 3 — Pricing engine (`pricing/`), validated against closed-form Black-Scholes
- [x] Phase 4 — Decision layer (`decision/`)
- [x] Phase 5 — Risk overlay (`risk/`)
- [x] Phase 6 — Backtesting (`backtest/`), walk-forward + placebo test
- [x] Phase 7 — Broker integration (`execution/`), Tradier sandbox paper trading
- [x] Phase 8 — Dashboard (`dashboard/`)
- [x] Phase 9 — Deployment (scheduling, logging, alerting)
- [x] `main.py` — wires all of the above into one end-to-end run

All 203 unit tests pass (`pytest`), including the Monte Carlo engine's
validation against closed-form Black-Scholes and the walk-forward/placebo
backtest mechanics. **None of this has been run against live data or a real
Tradier sandbox account yet** — every test above mocks the network. Before
trusting any of it:

1. Fill in real API keys in `.env` and confirm the data clients actually
   return sane data for a real ticker.
2. Wire the backtest engine (`backtest/`) to real historical chains via
   `data/polygon_client.py` and run it over real history before ever
   passing `--live` to `main.py`.

Since the first pass, three of the original simplifications have been
wired up for real (with tests, still against mocked data):
- **Dividend yield** is now estimated from trailing-twelve-month dividends
  (`data/yfinance_client.py:estimate_dividend_yield`), not hardcoded to 0.
- **Live position sizing** uses `risk/sizing.py`'s vol-targeted sizing,
  driven by an IV-vol-of-vol estimate built from `vol_richness.py`'s
  accumulated IV history (`factors/vol_richness.py:iv_vol_of_vol`) — it
  falls back to 1 contract, with a logged warning, only until a ticker has
  enough IV history (10+ observations) accumulated.
- **Beta-weighted delta** now threads the real rolling beta through to the
  live order path (it was computed but silently dropped before); when
  there isn't enough history for a ticker, the 1.0 fallback fires with a
  logged warning rather than silently.

`main.py --loop` re-runs the pipeline on an interval during market hours
(America/New_York, configurable in `config.py`) and appends every run to
`logs/candidates_history.jsonl` rather than only overwriting the latest
snapshot — intended for a scheduler (see `deployment/scheduling.md`), not
for running unattended in a foreground shell indefinitely.

Even with `--live`, `execution/tradier_broker.py` only talks to Tradier's
*sandbox* (paper trading) endpoint — nothing in this codebase places a
real-money order.

## Setup

```
cp .env.example .env
# fill in real values in .env (never commit it)
pip install -r requirements.txt
```

Required API keys (see `.env.example`):

| Variable | Source | Purpose |
|---|---|---|
| `POLYGON_API_KEY` | [polygon.io](https://polygon.io) (Options Starter/Developer tier) | Options chains, reference data, historical per-contract OHLC |
| `TRADIER_API_KEY` / `TRADIER_ACCOUNT_ID` | [tradier.com](https://tradier.com) (sandbox account) | Market-data IV/Greeks cross-check; paper-trading broker execution |
| `FRED_API_KEY` | [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) | Risk-free rate (`DGS3MO`, 3-month Treasury yield) |

Underlying price/dividend history uses `yfinance`, which needs no key.

Confirm current Polygon/Tradier plan pricing and rate limits directly with
each vendor before relying on this — they change.

## Repo layout

```
data/              ingestion + local caching (SQLite/Parquet) for chains, price history, rates
factors/           one module per factor, each independently testable/backtestable
pricing/           distribution construction + Monte Carlo/Black-Scholes pricer
decision/          edge calculation, ranking, trade selection
risk/              sizing, portfolio Greek/beta limits, delta-hedging
backtest/          walk-forward engine, cost modeling, placebo tests
execution/         broker integration, order lifecycle, reconciliation
dashboard/         read-only Streamlit app
tests/             mirrors the structure above
config.py          tunables and thresholds, no secrets
.env.example       documents required env vars with placeholder values
main.py            orchestrates a full run end to end
```
