# Pairs Trading Backtester — A Statistical Arbitrage Study

An event-driven backtesting engine built from scratch, used to test a
statistical-arbitrage (pairs trading) strategy with full rigor: no lookahead,
realistic transaction costs, out-of-sample validation, and honest reporting.

## Headline result

**The strategy has no tradeable edge — and that is the finding.**

The most coherent candidate pair (Coca-Cola / PepsiCo) showed a marginal
*in-sample* edge (Sharpe 0.20) that **collapsed to negative out-of-sample**
(Sharpe −0.50) on data the strategy never saw. Kelly position sizing, trusting
the flawed in-sample estimate, *amplified* the out-of-sample loss (−1.29% → −3.62%,
drawdown −2.01% → −5.83%). A robustness sweep confirmed the null across every
reasonable parameter setting.

This project demonstrates the full quantitative process working correctly,
including the part most backtests skip: testing honestly and reporting that the
edge isn't real. Consistent with market efficiency, the textbook pairs are too
efficiently arbitraged to yield an edge at daily frequency after costs.

## Why this project

I study data science with a math minor; I don't take finance courses. Building
this was how I learned the practical machinery of quant finance — risk, edge
estimation, and position sizing — extending a poker bot I wrote earlier (the
same core idea: find an edge, estimate it, size the bet accordingly).

## What it does

- Downloads and cleans adjusted daily price data (dividends/splits handled), cached locally.
- Runs an **event-driven backtest** whose architecture makes lookahead bias
  structurally impossible.
- Models realistic execution: commission, slippage, and **next-bar fills**.
- Screens pairs for tradeability with **cointegration** tests (Engle-Granger + ADF).
- Trades a mean-reverting spread via **rolling z-score** signals.
- Sizes positions with the **Kelly criterion** (fractional, capped).
- Validates strictly **out-of-sample** with a **robustness** sweep.

## Engine architecture

The backtest is event-driven: it walks through time one bar at a time and passes
events down a strict one-way chain:

    MarketEvent -> SignalEvent -> OrderEvent -> FillEvent

Information only ever flows forward. The data handler exposes prices through a
cursor that slices history at "today," so a strategy **cannot** read future
prices even by mistake. This is the core design principle — the difference
between a backtest you *hope* doesn't cheat and one that *can't*.

Realism layers on top: orders decided on day *t* fill at day *t+1*'s price
(next-bar execution, avoiding the "trade at the price you just saw" bias), with
commission (per-share, $1 min) and slippage (5 bps, always adverse).

## Methodology

1. **Data** — 10 tickers, 2015–present, adjusted daily closes. Train/test split
   at 2021-12-31; 2022+ held out and never used for fitting.
2. **Pair selection** — cointegration screen on the training window. Of five
   economically-motivated pairs, only V/MA (p=0.007) and KO/PEP (p=0.024) were
   cointegrated. The famous EWA/EWC textbook pair failed (p=0.23).
3. **Edge screen** — a cointegrated pair is necessary but *not sufficient*. V/MA,
   the cleanest cointegration, had essentially zero gross edge (Sharpe 0.05) —
   it is too efficiently arbitraged. KO/PEP was the only pair both cointegrated
   and net-positive in-sample, so it was carried forward.
4. **Signal** — spread = KO − 0.289·PEP (static hedge ratio from training).
   Rolling z-score; enter at |z|>2, exit at |z|<0.5, stop at |z|>3.5.
5. **Sizing** — continuous Kelly f* = μ/σ², half-Kelly, capped at 3x leverage.
6. **Validation** — freeze all parameters from training; run on unseen 2022+ data.

## Results (KO/PEP)

|                  | Return | Sharpe | Max DD |
|------------------|-------:|-------:|-------:|
| In-sample base   |  0.49% |   0.20 | −0.65% |
| In-sample Kelly  |  1.82% |   0.26 | −1.93% |
| Out-sample base  | −1.29% |  −0.50 | −2.01% |
| Out-sample Kelly | −3.62% |  −0.46 | −5.83% |

The in-sample edge was marginal and did not survive out-of-sample. A parameter
robustness grid (lookback × entry threshold) was flat-to-negative in 14 of 16
cells out-of-sample; the two positive cells were isolated, at the sparsest-
trading corner — textbook multiple-testing noise, not a real edge.

## Key lessons demonstrated

- **Cointegration ≠ tradeable edge.** V/MA was the most cointegrated pair and had
  the *least* edge — the cleaner and more obvious the relationship, the more
  thoroughly it's already arbitraged away.
- **The highest Sharpe can be a trap.** EWA/EWC had the best gross Sharpe (0.54)
  but wasn't cointegrated — a spurious pattern the statistical filter correctly
  rejected.
- **In-sample results lie.** A +0.20 in-sample Sharpe became −0.50 out-of-sample.
- **Kelly amplifies mis-estimated edges.** Levering a flawed edge tripled the
  out-of-sample loss. On low-variance strategies Kelly's leverage suggestion
  explodes (raw f* ≈ 59 here), so the cap — not the formula — becomes the real
  risk control.
- **Don't data-mine.** Picking the best of many pairs manufactures spurious edges;
  one candidate was chosen on principle and tested honestly.

## Limitations

Daily close data only; no intraday or order-book dynamics. Short-borrow fees and
margin are not modeled. Cost assumptions are conservative estimates, not a
specific broker's schedule. The universe is small and US-large-cap focused.

## Project structure

    data/         data download, cleaning, caching
    engine/       event classes, data handler, portfolio, execution, backtest loop
    strategies/   buy-and-hold, MA crossover (engine tests), pairs trading
    analysis/     cointegration, performance metrics, Kelly sizing
    scripts/      runnable analyses (screening, diagnostics, out-of-sample, robustness)

## Running it

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

    python -m scripts.download_data          # fetch + cache data
    python -m scripts.analyze_cointegration  # which pairs are tradeable?
    python -m scripts.screen_edge            # gross vs net edge per pair
    python -m scripts.out_of_sample          # the reckoning (in- vs out-of-sample)
    python -m scripts.robustness             # parameter robustness grid

## Tech stack

Python, pandas, NumPy, statsmodels (cointegration/ADF/OLS), yfinance, pyarrow.
