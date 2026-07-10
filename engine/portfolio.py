"""Portfolio: converts signals into orders and applies fills."""
from collections import defaultdict
from engine.events import OrderEvent


class Portfolio:
    def __init__(self, data, initial_capital=100_000, order_size=100):
        self.data = data
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.order_size = order_size
        self.positions = defaultdict(int)
        self.equity_curve = []

    def on_signal(self, event, data, events):
        current = self.positions[event.symbol]
        # Pairs strategies set target_quantity explicitly (each leg sized by the
        # hedge ratio). Simple strategies leave it None -> direction + order_size.
        if event.target_quantity is not None:
            target = event.target_quantity
        elif event.direction == "LONG":
            target = self.order_size
        elif event.direction == "SHORT":
            target = -self.order_size
        else:
            target = 0
        quantity = target - current           # trade only the delta to the target
        if quantity != 0:
            events.append(OrderEvent(event.timestamp, event.symbol, quantity))

    def on_fill(self, event):
        self.positions[event.symbol] += event.quantity
        self.cash -= event.quantity * event.fill_price   # short (neg qty) ADDS cash
        self.cash -= event.commission

    def mark_to_market(self, data):
        total = self.cash
        for symbol, qty in self.positions.items():
            price = data.current_price(symbol)
            if price is not None:
                total += qty * price
        self.equity_curve.append((data.current_timestamp(), total))
