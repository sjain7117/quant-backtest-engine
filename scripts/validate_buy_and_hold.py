"""Phase 2 quality-control gate: does buy-and-hold reproduce the asset return?

We run buy-and-hold through the engine, then compare the engine's OWN final
account value (from its mark-to-market equity curve) against a value we compute
by hand. If the engine's cash + position accounting is honest, they match to
floating-point precision. This tests the bookkeeping, not just the algebra.
"""
from engine.data_handler import PairDataHandler
from engine.portfolio import Portfolio
from engine.execution import SimulatedExecution
from engine.backtest import Backtest
from strategies.buy_and_hold import BuyAndHold

SYMBOL = "KO"

data = PairDataHandler(SYMBOL, "PEP")   # PEP just rides along; we only trade KO
strategy = BuyAndHold(SYMBOL)
portfolio = Portfolio(data, initial_capital=100_000, order_size=100)
execution = SimulatedExecution()

bt = Backtest(data, strategy, portfolio, execution)
bt.run()

# --- Ground-truth inputs from the raw price series ---
shares = portfolio.order_size
initial = portfolio.initial_capital
entry_price = data.prices[SYMBOL].iloc[0]     # close on the first bar (our entry)
final_price = data.prices[SYMBOL].iloc[-1]    # close on the last bar

# --- Engine's own number vs. hand-computed expectation ---
engine_final = portfolio.equity_curve[-1][1]                    # what the engine says
expected_final = initial + shares * (final_price - entry_price) # what it SHOULD be

# --- The conceptual point: the position's return == the asset's price return ---
asset_return = final_price / entry_price - 1

print(f"entry price        : {entry_price:.4f}")
print(f"final price        : {final_price:.4f}")
print(f"engine final value : {engine_final:,.2f}")
print(f"expected final val : {expected_final:,.2f}")
print(f"difference         : {abs(engine_final - expected_final):.2e}")
print(f"KO asset return    : {asset_return:.4%}")
print()
print("BOOKKEEPING:", "PASS" if abs(engine_final - expected_final) < 1e-6 else "FAIL")
