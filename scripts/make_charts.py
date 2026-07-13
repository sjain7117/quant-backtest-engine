"""Generate the README chart: mean-reversion vs. momentum, mirror-image panels.

Each strategy runs over the full period with parameters frozen at the train/test
split (dashed line). Both equity curves are indexed to 100 ($100 growth) so the
SHAPES are comparable despite very different position sizes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from data.loader import download_prices, CANDIDATE_UNIVERSE
from analysis.cointegration import hedge_ratio
from engine.data_handler import PairDataHandler, MultiDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum import CrossSectionalMomentum
from analysis.performance import equity_curve_to_series

TRAIN_END = pd.Timestamp("2021-12-31")
symbols = list(CANDIDATE_UNIVERSE)
prices = download_prices(symbols).dropna()


def costs():
    return SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))


ko_pep = prices[["KO", "PEP"]].dropna()
beta, _, _ = hedge_ratio(ko_pep.loc[:TRAIN_END]["KO"], ko_pep.loc[:TRAIN_END]["PEP"])


def run_pairs():
    data = PairDataHandler("KO", "PEP", prices=ko_pep)
    strat = PairsTradingStrategy("KO", "PEP", beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=100)
    port = Portfolio(data, 100_000)
    Backtest(data, strat, port, costs()).run()
    return equity_curve_to_series(port.equity_curve)


def run_mom():
    data = MultiDataHandler(symbols, prices=prices)
    strat = CrossSectionalMomentum(symbols, lookback=126, skip=21,
                                   n_long=3, n_short=3, gross_per_side=50_000)
    port = Portfolio(data, 100_000)
    Backtest(data, strat, port, costs()).run()
    return equity_curve_to_series(port.equity_curve)


pairs = run_pairs(); pairs = pairs / pairs.iloc[0] * 100
mom = run_mom(); mom = mom / mom.iloc[0] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

ax1.plot(pairs.index, pairs.values, color="tab:blue", lw=1.4)
ax1.axvline(TRAIN_END, color="k", ls="--", lw=1, alpha=0.7)
ax1.axhline(100, color="gray", ls=":", lw=1, alpha=0.6)
ax1.annotate("train | out-of-sample", xy=(TRAIN_END, ax1.get_ylim()[1]),
             xytext=(6, -12), textcoords="offset points", fontsize=9, va="top")
ax1.set_ylabel("Growth of $100")
ax1.set_title("Pairs (mean-reversion): faint edge before the split, inverts after")
ax1.grid(alpha=0.2)

ax2.plot(mom.index, mom.values, color="tab:orange", lw=1.4)
ax2.axvline(TRAIN_END, color="k", ls="--", lw=1, alpha=0.7)
ax2.axhline(100, color="gray", ls=":", lw=1, alpha=0.6)
ax2.set_ylabel("Growth of $100")
ax2.set_title("Momentum (trend): flat before the split, positive after")
ax2.grid(alpha=0.2)

plt.tight_layout()
Path("assets").mkdir(exist_ok=True)
fig.savefig("assets/performance.png", dpi=130, bbox_inches="tight")
print("saved assets/performance.png")
