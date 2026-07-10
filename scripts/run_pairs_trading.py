"""Phase 4b: run the V/MA pairs strategy IN-SAMPLE (training window) as a sanity
check that the signal logic trades sensibly.

This is NOT a result to believe: in-sample performance is optimistic because the
hedge ratio and the whole period were part of what we fit on. The honest
out-of-sample test comes in Phase 5.
"""
from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics, print_report

SYM_A, SYM_B = "V", "MA"
TRAIN_END = "2021-12-31"

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]

# Static hedge ratio estimated on TRAINING data only.
beta, alpha, _ = hedge_ratio(train[SYM_A], train[SYM_B])
print(f"Static hedge ratio (beta) from training: {beta:.4f}")
print(f"Spread definition: {SYM_A} - {beta:.4f} * {SYM_B}\n")

data = PairDataHandler(SYM_A, SYM_B, prices=train)          # in-sample run
strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=30,
                             entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=100)
port = Portfolio(data, initial_capital=100_000)
execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
counts = Backtest(data, strat, port, execu).run()

print(f"Fills: {counts['FILL']}  (each entry or exit moves BOTH legs)\n")
eq = equity_curve_to_series(port.equity_curve)
print_report(compute_metrics(eq), "V/MA PAIRS -- IN-SAMPLE (training window, optimistic)")

print("\nThis is IN-SAMPLE: beta and the whole period were seen during fitting.")
print("Believe nothing yet -- Phase 5 runs this out-of-sample on 2022+ data.")
