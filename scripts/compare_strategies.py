"""Head-to-head: mean-reversion (pairs) vs momentum (trend).

Same engine, same costs, same train/test split. Compare on SHARPE (scale-
invariant), since the two strategies use different position sizes. Parameters are
the ones chosen before any robustness grid, so nothing is cherry-picked.
"""
import pandas as pd
from data.loader import download_prices, CANDIDATE_UNIVERSE
from analysis.cointegration import hedge_ratio
from engine.data_handler import PairDataHandler, MultiDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, PerShareCommission, BpsSlippage
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from strategies.momentum import CrossSectionalMomentum
from analysis.performance import equity_curve_to_series, compute_metrics

TRAIN_END, TEST_START = "2021-12-31", "2022-01-01"
symbols = list(CANDIDATE_UNIVERSE)
prices = download_prices(symbols).dropna()


def costs():
    return SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))


def metrics_of(port, metrics_from):
    eq = equity_curve_to_series(port.equity_curve)
    if metrics_from is not None:
        eq = eq[eq.index >= pd.Timestamp(metrics_from)]
    return compute_metrics(eq)


ko_pep = prices[["KO", "PEP"]].dropna()
beta, _, _ = hedge_ratio(ko_pep.loc[:TRAIN_END]["KO"], ko_pep.loc[:TRAIN_END]["PEP"])
fp = int((ko_pep.index >= pd.Timestamp(TEST_START)).argmax())


def run_pairs(frame, metrics_from=None):
    data = PairDataHandler("KO", "PEP", prices=frame)
    strat = PairsTradingStrategy("KO", "PEP", beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=100)
    port = Portfolio(data, 100_000)
    Backtest(data, strat, port, costs()).run()
    return metrics_of(port, metrics_from)


fm = int((prices.index >= pd.Timestamp(TEST_START)).argmax())


def run_mom(frame, metrics_from=None):
    data = MultiDataHandler(symbols, prices=frame)
    strat = CrossSectionalMomentum(symbols, lookback=126, skip=21,
                                   n_long=3, n_short=3, gross_per_side=50_000)
    port = Portfolio(data, 100_000)
    Backtest(data, strat, port, costs()).run()
    return metrics_of(port, metrics_from)


rows = [
    ("Pairs (mean-rev)", "in-sample  2015-21", run_pairs(ko_pep.loc[:TRAIN_END])),
    ("Pairs (mean-rev)", "out-sample 2022+", run_pairs(ko_pep.iloc[max(0, fp-60):], TEST_START)),
    ("Momentum (trend)", "in-sample  2015-21", run_mom(prices.loc[:TRAIN_END])),
    ("Momentum (trend)", "out-sample 2022+", run_mom(prices.iloc[max(0, fm-300):], TEST_START)),
]

print(f"{'strategy':<18}{'period':<20}{'return':>9}{'Sharpe':>8}{'maxDD':>8}")
print("-" * 63)
for name, period, m in rows:
    print(f"{name:<18}{period:<20}{m['total_return']:>9.2%}{m['sharpe']:>8.2f}{m['max_drawdown']:>8.2%}")

print("\nCompare on SHARPE (scale-invariant). Mirror-image pattern:")
print("pairs works in-sample then dies; momentum does nothing then works.")
