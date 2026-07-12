"""Generate the README performance chart: equity curves + spread z-score.

Runs the frozen-parameter KO/PEP strategy over the FULL period (train + test),
draws base vs Kelly equity with a train/test split line, and the spread z-score
with entry/exit bands. Saves assets/performance.png for the README.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from analysis.kelly import kelly_sized_units
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series

SYM_A, SYM_B = "KO", "PEP"
TRAIN_END = pd.Timestamp("2021-12-31")
BASE_UNITS, LOOKBACK = 100, 30

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]
beta, _, _ = hedge_ratio(train[SYM_A], train[SYM_B])


def run(units, price_frame):
    data = PairDataHandler(SYM_A, SYM_B, prices=price_frame)
    strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=LOOKBACK,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=units)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    return equity_curve_to_series(port.equity_curve)


kelly_units, _, _ = kelly_sized_units(BASE_UNITS, run(BASE_UNITS, train).pct_change().dropna())
eq_base = run(BASE_UNITS, prices)
eq_kelly = run(kelly_units, prices)

spread = prices[SYM_A] - beta * prices[SYM_B]
z = (spread - spread.rolling(LOOKBACK).mean()) / spread.rolling(LOOKBACK).std()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                               gridspec_kw={"height_ratios": [2, 1]}, sharex=True)

ax1.plot(eq_base.index, eq_base.values, label=f"Base ({BASE_UNITS} units)", lw=1.4)
ax1.plot(eq_kelly.index, eq_kelly.values, label=f"Kelly ({kelly_units} units)", lw=1.4)
ax1.axvline(TRAIN_END, color="k", ls="--", lw=1, alpha=0.7)
ax1.axhline(100_000, color="gray", ls=":", lw=1, alpha=0.6)
ax1.annotate("train | out-of-sample", xy=(TRAIN_END, ax1.get_ylim()[1]),
             xytext=(6, -12), textcoords="offset points", fontsize=9, va="top")
ax1.set_ylabel("Account value ($)")
ax1.set_title("KO/PEP pairs strategy — equity (parameters frozen at the train/test split)")
ax1.legend(loc="upper left"); ax1.grid(alpha=0.2)

ax2.plot(z.index, z.values, lw=0.7, color="tab:purple")
for lvl in (2, -2):
    ax2.axhline(lvl, color="tab:red", ls="--", lw=0.8, alpha=0.7)
for lvl in (0.5, -0.5):
    ax2.axhline(lvl, color="tab:green", ls=":", lw=0.8, alpha=0.7)
ax2.axvline(TRAIN_END, color="k", ls="--", lw=1, alpha=0.7)
ax2.set_ylabel("Spread z-score")
ax2.set_title("Spread z-score with entry (±2, red) and exit (±0.5, green) bands")
ax2.grid(alpha=0.2)

plt.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/performance.png", dpi=130, bbox_inches="tight")
print("saved assets/performance.png")
