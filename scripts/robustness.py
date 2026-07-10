"""Phase 5 robustness: does the negative out-of-sample verdict hold across
reasonable parameter choices, or did we just pick one unlucky setting?

We sweep lookback x entry_z and report OUT-OF-SAMPLE Sharpe for each cell. If
the whole grid sits around zero or negative, 'no edge' is robust -- not an
artifact of one choice.

Note the direction of reasoning: we sweep to CONFIRM a null (show nothing
works), NOT to hunt for a winner. Finding one green cell in a big grid would
just be the multiple-testing trap again.
"""
import pandas as pd
from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics

SYM_A, SYM_B = "KO", "PEP"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
WARMUP = 80
BASE_UNITS = 100
LOOKBACKS = [20, 30, 45, 60]
ENTRIES = [1.5, 2.0, 2.5, 3.0]

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]
beta, _, _ = hedge_ratio(train[SYM_A], train[SYM_B])

first_test_pos = int((prices.index >= pd.Timestamp(TEST_START)).argmax())
test_slice = prices.iloc[max(0, first_test_pos - WARMUP):]


def oos_sharpe(lookback, entry_z):
    data = PairDataHandler(SYM_A, SYM_B, prices=test_slice)
    strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=lookback,
                                 entry_z=entry_z, exit_z=0.5, stop_z=3.5, base_units=BASE_UNITS)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    eq = equity_curve_to_series(port.equity_curve)
    eq = eq[eq.index >= pd.Timestamp(TEST_START)]
    m = compute_metrics(eq)
    return m.get("sharpe", float("nan"))


print(f"OUT-OF-SAMPLE Sharpe grid (beta={beta:.3f}, base sizing)\n")
print("lookback \\ entry_z" + "".join(f"{e:>8}" for e in ENTRIES))
print("-" * (18 + 8 * len(ENTRIES)))
for lb in LOOKBACKS:
    row = "".join(f"{oos_sharpe(lb, e):>8.2f}" for e in ENTRIES)
    print(f"{lb:<18}{row}")

print("\nIf the grid is uniformly around zero or negative, the 'no tradeable edge'")
print("conclusion is robust to parameter choice -- the honest, bulletproof verdict.")
