"""Accounting gate, updated for Phase 3's next-bar execution.

Uses ZERO-COST execution so this stays a pure bookkeeping test. With next-bar
execution the entry fills on bar index 1 (the bar AFTER the signal), so the
hand-computed expectation uses prices.iloc[1] as the entry price.
"""
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution, NoCommission, NoSlippage
from engine.backtest import Backtest
from strategies.buy_and_hold import BuyAndHold

SYMBOL = "KO"

data = PairDataHandler(SYMBOL, "PEP")
strategy = BuyAndHold(SYMBOL)
portfolio = Portfolio(data, initial_capital=100_000, order_size=100)
execution = SimulatedExecution(NoCommission(), NoSlippage())   # frictionless on purpose

bt = Backtest(data, strategy, portfolio, execution)
bt.run()

shares = portfolio.order_size
initial = portfolio.initial_capital
entry_price = data.prices[SYMBOL].iloc[1]     # next-bar fill: signal bar 0, fill bar 1
final_price = data.prices[SYMBOL].iloc[-1]

engine_final = portfolio.equity_curve[-1][1]
expected_final = initial + shares * (final_price - entry_price)

print(f"entry price        : {entry_price:.4f}")
print(f"final price        : {final_price:.4f}")
print(f"engine final value : {engine_final:,.2f}")
print(f"expected final val : {expected_final:,.2f}")
print(f"difference         : {abs(engine_final - expected_final):.2e}")
print()
print("BOOKKEEPING:", "PASS" if abs(engine_final - expected_final) < 1e-6 else "FAIL")
