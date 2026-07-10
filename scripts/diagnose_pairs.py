"""Phase 4b diagnostic: WHY did the pairs strategy lose in-sample?

Two questions:
  1. Is there a GROSS edge? -> run frictionless vs. with-costs.
     If frictionless is positive but costs flip it negative, the edge is real
     but too thin to beat the 'rake'. If frictionless is also flat/negative,
     the signal itself has no edge.
  2. Does trading LESS (higher entry threshold -> fewer, higher-conviction
     trades) let the edge clear costs?
"""
from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import (SimulatedExecution, PerShareCommission,
                              BpsSlippage, NoCommission, NoSlippage)
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics

SYM_A, SYM_B = "V", "MA"
TRAIN_END = "2021-12-31"

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]
beta, _, _ = hedge_ratio(train[SYM_A], train[SYM_B])


def run(entry_z, with_costs, lookback=30):
    data = PairDataHandler(SYM_A, SYM_B, prices=train)
    strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=lookback,
                                 entry_z=entry_z, exit_z=0.5, stop_z=3.5, base_units=100)
    if with_costs:
        execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    else:
        execu = SimulatedExecution(NoCommission(), NoSlippage())
    port = Portfolio(data, initial_capital=100_000)
    counts = Backtest(data, strat, port, execu).run()
    m = compute_metrics(equity_curve_to_series(port.equity_curve))
    return counts["FILL"], m


print(f"beta = {beta:.4f}\n")
print("1) IS THERE A GROSS EDGE?  (entry_z = 2.0)")
print("-" * 52)
for label, costs in [("frictionless", False), ("with costs ", True)]:
    fills, m = run(2.0, costs)
    print(f"  {label}:  return {m['total_return']:>7.2%}   "
          f"Sharpe {m['sharpe']:>6.2f}   fills {fills}")

print("\n2) DOES TRADING LESS HELP?  (entry threshold sweep, WITH costs)")
print("-" * 52)
print(f"  {'entry_z':>8}{'fills':>8}{'return':>10}{'Sharpe':>9}")
for ez in [1.5, 2.0, 2.5, 3.0]:
    fills, m = run(ez, True)
    print(f"  {ez:>8.1f}{fills:>8}{m['total_return']:>10.2%}{m['sharpe']:>9.2f}")

print("\n  (frictionless edge at each threshold, for contrast)")
print(f"  {'entry_z':>8}{'fills':>8}{'return':>10}{'Sharpe':>9}")
for ez in [1.5, 2.0, 2.5, 3.0]:
    fills, m = run(ez, False)
    print(f"  {ez:>8.1f}{fills:>8}{m['total_return']:>10.2%}{m['sharpe']:>9.2f}")
