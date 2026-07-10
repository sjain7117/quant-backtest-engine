"""Phase 3 payoff: run the MA crossover frictionless vs. WITH realistic costs,
and print the full performance report for each, so the drag from trading is
explicit and measurable.
"""
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import (SimulatedExecution, PerShareCommission,
                              BpsSlippage, NoCommission, NoSlippage)
from engine.backtest import Backtest
from strategies.ma_crossover import MACrossover
from analysis.performance import (equity_curve_to_series, compute_metrics,
                                  print_report)

SYMBOL = "KO"


def run(execution, label):
    data = PairDataHandler(SYMBOL, "PEP")
    strat = MACrossover(SYMBOL, short_window=20, long_window=100)
    port = Portfolio(data, initial_capital=100_000, order_size=100)
    counts = Backtest(data, strat, port, execution).run()
    equity = equity_curve_to_series(port.equity_curve)
    metrics = compute_metrics(equity)
    print_report(metrics, f"{label}  ({counts['FILL']} fills)")
    print()
    return metrics


print("KO  MA crossover (20 / 100)\n")
m_free = run(SimulatedExecution(NoCommission(), NoSlippage()), "FRICTIONLESS")
m_cost = run(SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0)),
             "WITH COSTS  (5bps slippage, $0.005/share, $1 min)")

drag = m_free["total_return"] - m_cost["total_return"]
print(f"Return lost to trading costs: {drag:.2%}")
