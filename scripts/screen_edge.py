"""Screen candidate pairs for a GROSS tradeable edge.

Cointegration got us this far. This asks the harder question that actually
matters: does the spread yield a mean-reversion PROFIT -- before costs (is there
any edge at all?) and after costs (does it survive the rake)?

A pair can be strongly cointegrated yet have zero tradeable edge, because the
reversion isn't predictable/consistent enough, or because the pair is so
efficiently arbitraged that nothing is left at daily frequency.
"""
from data.loader import download_prices
from analysis.cointegration import hedge_ratio, engle_granger_pvalue
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import (SimulatedExecution, PerShareCommission,
                              BpsSlippage, NoCommission, NoSlippage)
from engine.backtest import Backtest
from strategies.pairs_trading import PairsTradingStrategy
from analysis.performance import equity_curve_to_series, compute_metrics

PAIRS = [("KO", "PEP"), ("GLD", "GDX"), ("XOM", "CVX"), ("V", "MA"), ("EWA", "EWC")]
TRAIN_END = "2021-12-31"


def run(prices, a, b, beta, with_costs):
    data = PairDataHandler(a, b, prices=prices)
    strat = PairsTradingStrategy(a, b, beta=beta, lookback=30,
                                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=100)
    execu = (SimulatedExecution(PerShareCommission(0.005, 1.0), BpsSlippage(5.0))
             if with_costs else SimulatedExecution(NoCommission(), NoSlippage()))
    port = Portfolio(data, initial_capital=100_000)
    Backtest(data, strat, port, execu).run()
    return compute_metrics(equity_curve_to_series(port.equity_curve))


tickers = sorted({t for p in PAIRS for t in p})
all_prices = download_prices(tickers, start="2015-01-01")

print("Screening candidate pairs for a tradeable edge (in-sample, 2015-2021)\n")
print(f"{'pair':<9}{'EG p':>8}{'  |  '}{'gross ret':>10}{'gross SR':>9}"
      f"{'  |  '}{'net ret':>9}{'net SR':>8}")
print("-" * 68)
for a, b in PAIRS:
    sub = all_prices[[a, b]].dropna()
    train = sub.loc[:TRAIN_END]
    beta, _, _ = hedge_ratio(train[a], train[b])
    _, eg_p = engle_granger_pvalue(train[a], train[b])
    gross = run(train, a, b, beta, with_costs=False)
    net = run(train, a, b, beta, with_costs=True)
    print(f"{a+'/'+b:<9}{eg_p:>8.3f}{'  |  '}"
          f"{gross['total_return']:>10.2%}{gross['sharpe']:>9.2f}{'  |  '}"
          f"{net['total_return']:>9.2%}{net['sharpe']:>8.2f}")

print("\nRead:  EG p < 0.05 = cointegrated.  gross = frictionless (is there ANY")
print("edge?).  net = after costs (does it survive?).  A Sharpe near 0 means no")
print("real edge regardless of the sign of the return.")
