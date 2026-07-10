"""Phase 5: the reckoning. Train params FROZEN; test on unseen 2022+ data.

Nothing is re-fit on the test period: beta, the Kelly sizing, and every
threshold come only from training (<=2021). We run the strategy on 2022+ -- data
it has never seen -- and compare in-sample vs out-of-sample. That gap is the
whole point: an edge that only exists in-sample was never real.

A short WARMUP buffer of pre-2022 bars is fed in so the rolling z-score is
already 'warm' on the first test day (using past data is not lookahead);
performance is measured only from TEST_START onward.
"""
import pandas as pd
from data.loader import download_prices
from analysis.cointegration import hedge_ratio
from analysis.kelly import kelly_sized_units
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics

SYM_A, SYM_B = "KO", "PEP"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
WARMUP = 60
BASE_UNITS = 100

prices = download_prices([SYM_A, SYM_B], start="2015-01-01").dropna()
train = prices.loc[:TRAIN_END]
beta, _, _ = hedge_ratio(train[SYM_A], train[SYM_B])


def run(prices_slice, units, metrics_from=None):
    data = PairDataHandler(SYM_A, SYM_B, prices=prices_slice)
    strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=units)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    eq = equity_curve_to_series(port.equity_curve)
    if metrics_from is not None:
        eq = eq[eq.index >= pd.Timestamp(metrics_from)]
    return compute_metrics(eq)


# Freeze Kelly sizing from a TRAIN base run.
data = PairDataHandler(SYM_A, SYM_B, prices=train)
strat = PairsTradingStrategy(SYM_A, SYM_B, beta=beta, lookback=30, base_units=BASE_UNITS)
execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
port = Portfolio(data, initial_capital=100_000)
Backtest(data, strat, port, execu).run()
train_returns = equity_curve_to_series(port.equity_curve).pct_change().dropna()
kelly_units, f_star, f_app = kelly_sized_units(BASE_UNITS, train_returns)

# Build the test slice WITH a warmup buffer before 2022.
first_test_pos = int((prices.index >= pd.Timestamp(TEST_START)).argmax())
warm_start = max(0, first_test_pos - WARMUP)
test_slice = prices.iloc[warm_start:]

print(f"beta (train) = {beta:.4f}   |   Kelly units (train) = {kelly_units}")
print(f"train: {train.index.min().date()} -> {train.index.max().date()}   "
      f"test: {pd.Timestamp(TEST_START).date()} -> {prices.index.max().date()}\n")

rows = [
    ("IN-SAMPLE  base",  run(train, BASE_UNITS)),
    ("IN-SAMPLE  Kelly", run(train, kelly_units)),
    ("OUT-SAMPLE base",  run(test_slice, BASE_UNITS, metrics_from=TEST_START)),
    ("OUT-SAMPLE Kelly", run(test_slice, kelly_units, metrics_from=TEST_START)),
]

print(f"{'':<18}{'return':>9}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}")
print("-" * 50)
for label, m in rows:
    print(f"{label:<18}{m['total_return']:>9.2%}{m['cagr']:>8.2%}"
          f"{m['sharpe']:>8.2f}{m['max_drawdown']:>8.2%}")
