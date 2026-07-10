"""Simulated broker with realistic frictions and NEXT-BAR execution.

Two costs are modeled:
  - commission: the fee paid per trade
  - slippage:   the small ADVERSE price move between deciding and executing
                (you rarely get the exact price you saw; buys fill a bit higher,
                 sells a bit lower)

And the key timing rule: an order decided on bar t fills on bar t+1, at that
bar's price. A strategy decides using today's close, so it must NOT also trade
AT today's close -- that would assume transacting at a price only known after
deciding. Next-bar execution removes that bias.
"""
from collections import deque
from engine.events import FillEvent


# ----- Commission models -------------------------------------------------
class PerShareCommission:
    """IB-style: a cost per share, with a per-order minimum."""
    def __init__(self, cost_per_share=0.005, minimum=1.0):
        self.cost_per_share = cost_per_share
        self.minimum = minimum

    def calculate(self, quantity, price):
        return max(self.minimum, abs(quantity) * self.cost_per_share)


class PercentageCommission:
    """Commission as basis points of traded notional (1 bp = 0.01%)."""
    def __init__(self, bps=1.0):
        self.rate = bps / 10_000.0

    def calculate(self, quantity, price):
        return abs(quantity) * price * self.rate


class NoCommission:
    """Frictionless commission -- used for pure accounting tests."""
    def calculate(self, quantity, price):
        return 0.0


# ----- Slippage models ---------------------------------------------------
class BpsSlippage:
    """Adverse price move in basis points; always works against you."""
    def __init__(self, bps=5.0):
        self.rate = bps / 10_000.0

    def apply(self, price, quantity):
        direction = 1 if quantity > 0 else -1   # buys fill higher, sells lower
        return price * (1 + direction * self.rate)


class NoSlippage:
    """Frictionless slippage -- used for pure accounting tests."""
    def apply(self, price, quantity):
        return price


# ----- The execution handler ---------------------------------------------
class SimulatedExecution:
    def __init__(self, commission_model=None, slippage_model=None):
        # Sensible realistic defaults; pass NoCommission()/NoSlippage() to disable.
        self.commission_model = commission_model or PerShareCommission()
        self.slippage_model = slippage_model or BpsSlippage()
        self.pending = deque()

    def on_order(self, event, data, events):
        """Queue the order -- do NOT fill now. It fills on the next bar."""
        self.pending.append(event)

    def execute_pending(self, data, events):
        """Fill every queued order at the CURRENT bar's price.

        Called once when a new bar arrives, BEFORE the strategy sees that bar,
        so orders decided last bar transact at this bar's price.
        """
        while self.pending:
            order = self.pending.popleft()
            base = data.current_price(order.symbol)
            fill_price = self.slippage_model.apply(base, order.quantity)
            commission = self.commission_model.calculate(order.quantity, fill_price)
            events.append(FillEvent(
                timestamp=data.current_timestamp(),
                symbol=order.symbol,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
            ))
