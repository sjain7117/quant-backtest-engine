"""Stress-test the momentum out-of-sample result across parameter choices.

Rebalancing is now fixed (monthly), so we sweep lookback x breadth (how many
names per side). Uniform sign supports a real regime effect; scattered signs
mean noise. As before, this interrogates a result -- it does not hunt for the
best cell.
"""
import pandas as pd
from data.loader import download_prices, CANDIDATE_UNIVERSE
from engine.data_handler import MultiDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.momentum import CrossSectionalMomentum
from analysis.performance import equity_curve_to_series, compute_metrics

TEST_START, WARMUP = "2022-01-01", 300
LOOKBACKS = [63, 126, 189, 252]
BREADTHS = [2, 3, 4]

symbols = list(CANDIDATE_UNIVERSE)
prices = download_prices(symbols).dropna()
first = int((prices.index >= pd.Timestamp(TEST_START)).argmax())
test_slice = prices.iloc[max(0, first - WARMUP):]


def oos_sharpe(lookback, breadth):
    data = MultiDataHandler(symbols, prices=test_slice)
    strat = CrossSectionalMomentum(symbols, lookback=lookback, skip=21,
                                   n_long=breadth, n_short=breadth, gross_per_side=50_000)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    eq = equity_curve_to_series(port.equity_curve)
    eq = eq[eq.index >= pd.Timestamp(TEST_START)]
    return compute_metrics(eq).get("sharpe", float("nan"))


print("OUT-OF-SAMPLE momentum Sharpe grid (2022+), monthly rebalance\n")
print("lookback \\ names/side" + "".join(f"{b:>8}" for b in BREADTHS))
print("-" * (21 + 8 * len(BREADTHS)))
for lb in LOOKBACKS:
    row = "".join(f"{oos_sharpe(lb, b):>8.2f}" for b in BREADTHS)
    print(f"{lb:<21}{row}")

print("\nUniform sign -> a real (regime-limited) effect; scattered -> noise.")
