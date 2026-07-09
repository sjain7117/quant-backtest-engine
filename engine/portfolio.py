"""Portfolio: the accountant. Converts signals into orders and applies fills.

Two responsibilities:
  1. on_signal -> decide the order needed to reach the target position
  2. on_fill   -> update positions and cash once an order executes
"""
from collections import defaultdict
from engine.events import OrderEvent


class Portfolio:
    def __init__(self, data, initial_capital=100_000, order_size=100):
        self.data = data
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.order_size = order_size            # fixed share count for now
        self.positions = defaultdict(int)       # symbol -> signed shares held
        self.equity_curve = []                  # (timestamp, total_value) per bar

    def on_signal(self, event, data, events):
        """Translate a desired direction into a SIGNED order quantity.

        We compute the order as (target position - current position). This
        'delta' approach means we never accidentally stack a second position on
        top of an existing one -- we only ever trade the difference needed.
        """
        current = self.positions[event.symbol]
        if event.direction == "LONG":
            target = self.order_size
        elif event.direction == "SHORT":
            target = -self.order_size
        else:  # "EXIT" -> flat
            target = 0

        quantity = target - current
        if quantity != 0:
            events.append(OrderEvent(event.timestamp, event.symbol, quantity))

    def on_fill(self, event):
        """Apply an executed trade: adjust shares, pay cash, pay commission."""
        self.positions[event.symbol] += event.quantity
        self.cash -= event.quantity * event.fill_price   # signed: buy costs, sell adds
        self.cash -= event.commission

    def mark_to_market(self, data):
        """Snapshot total account value (cash + shares valued at today's price)."""
        total = self.cash
        for symbol, qty in self.positions.items():
            price = data.current_price(symbol)
            if price is not None:
                total += qty * price
        self.equity_curve.append((data.current_timestamp(), total))
