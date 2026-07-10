"""Phase 4c: Kelly position sizing for the KO/PEP pairs strategy (in-sample).

Estimate the edge from a BASE run on training data, compute fractional-Kelly
sizing, then re-run with it. Everything is estimated on training only, so the
sizing (like beta) carries unchanged to the out-of-sample test in Phase 5.

Kelly sizes the BET; it does NOT change the Sharpe ratio. Watch return and
drawdown scale together while risk-adjusted return stays about the same.
"""
from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from analysis.kelly import kelly_sized_units
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics, print_report

SYM_A, SYM_B = "KO", "PEP"
TRAIN_END = "2021-12-31"
BASE_UNITS = 100

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]
beta, _, _ = hedge_ratio(train[SYM_A], train[SYM_B])


def run(units):
    data = PairDataHandler(SYM_A, SYM_B, prices=train)
    strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=units)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    eq = equity_curve_to_series(port.equity_curve)
    return eq, compute_metrics(eq)


base_eq, base_m = run(BASE_UNITS)
base_returns = base_eq.pct_change().dropna()

units, f_star, f_applied = kelly_sized_units(BASE_UNITS, base_returns,
                                             fraction=0.5, max_leverage=3.0)
print(f"beta = {beta:.4f}")
print(f"Full Kelly f*         : {f_star:>7.2f}")
print(f"Half-Kelly, capped 3x : {f_applied:>7.2f}")
print(f"Base -> Kelly units   : {BASE_UNITS} -> {units}\n")

print_report(base_m, f"BASE sizing ({BASE_UNITS} units) -- in-sample")
print()
if units > 0:
    _, kelly_m = run(units)
    print_report(kelly_m, f"KELLY sizing ({units} units) -- in-sample")
    print(f"\nSharpe: {base_m['sharpe']:.2f} -> {kelly_m['sharpe']:.2f}  "
          f"(barely moves -- Kelly rescaled the BET, not the edge)")
else:
    print("Kelly sized to 0 units: estimated edge does not justify trading.")

print(f"\nKelly units to carry OUT-OF-SAMPLE (fixed from training): {units}")
