"""Cross-sectional momentum over the SAME train/test periods as the pairs
strategy, so the two archetypes can be compared directly. Calendar-anchored
monthly rebalancing; a 300-bar warmup ensures the book is fully spun up before
the test period begins.
"""
import pandas as pd
from data.loader import download_prices, CANDIDATE_UNIVERSE
from engine.data_handler import MultiDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.momentum import CrossSectionalMomentum
from analysis.performance import equity_curve_to_series, compute_metrics, print_report

TRAIN_END, TEST_START, WARMUP = "2021-12-31", "2022-01-01", 300
symbols = list(CANDIDATE_UNIVERSE)
prices = download_prices(symbols).dropna()
train = prices.loc[:TRAIN_END]
first = int((prices.index >= pd.Timestamp(TEST_START)).argmax())
test_slice = prices.iloc[max(0, first - WARMUP):]


def run(price_frame, metrics_from=None):
    data = MultiDataHandler(symbols, prices=price_frame)
    strat = CrossSectionalMomentum(symbols, lookback=126, skip=21,
                                   n_long=3, n_short=3, gross_per_side=50_000)
    execu = SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    eq = equity_curve_to_series(port.equity_curve)
    if metrics_from is not None:
        eq = eq[eq.index >= pd.Timestamp(metrics_from)]
    return compute_metrics(eq)


print("Cross-sectional momentum (lookback 126, skip 21, monthly, 3 long / 3 short)\n")
print_report(run(train), "IN-SAMPLE 2015-2021")
print()
print_report(run(test_slice, metrics_from=TEST_START), "OUT-OF-SAMPLE 2022+")
