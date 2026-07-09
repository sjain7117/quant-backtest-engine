"""Run the MA crossover and compute performance off the equity curve.

This is the first time we turn the equity curve into RETURNS. We compute both
kinds side by side to make the distinction concrete:
  simple return  = pct change; describes a single period; COMPOUNDS (multiply)
  log return     = ln(ratio);  ADDS across periods; cleaner for aggregation
"""
import numpy as np
import pandas as pd

from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution
from engine.backtest import Backtest
from strategies.ma_crossover import MACrossover

SYMBOL = "KO"

data = PairDataHandler(SYMBOL, "PEP")
strategy = MACrossover(SYMBOL, short_window=20, long_window=100)
portfolio = Portfolio(data, initial_capital=100_000, order_size=100)
execution = SimulatedExecution()

bt = Backtest(data, strategy, portfolio, execution)
counts = bt.run()

# Equity curve -> a pandas Series of daily account value.
equity = pd.Series(
    [v for _, v in portfolio.equity_curve],
    index=[t for t, _ in portfolio.equity_curve],
)

# Two ways to measure daily change.
simple_returns = equity.pct_change().dropna()
log_returns = np.log(equity / equity.shift(1)).dropna()

# The key identity: summing log returns == log of the total growth factor.
total_growth = equity.iloc[-1] / equity.iloc[0]
sum_log = log_returns.sum()

print(f"number of trades   : {counts['FILL']}")
print(f"start value        : {equity.iloc[0]:,.2f}")
print(f"final value        : {equity.iloc[-1]:,.2f}")
print(f"total return       : {total_growth - 1:.4%}")
print()
print(f"sum of log returns : {sum_log:.6f}")
print(f"ln(total growth)   : {np.log(total_growth):.6f}   <- should match above")
print()
print(f"mean daily simple  : {simple_returns.mean():.6%}")
print(f"mean daily log     : {log_returns.mean():.6%}")
