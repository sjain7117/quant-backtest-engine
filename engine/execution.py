"""Execution handler: the simulated broker. Turns orders into fills.

For now it fills instantly at the current bar's closing price with zero cost.
Phase 3 adds the realism: commission and slippage (the small adverse price
move between deciding to trade and the trade actually happening).
"""
from engine.events import FillEvent


class SimulatedExecution:
    def on_order(self, event, data, events):
        price = data.current_price(event.symbol)
        fill = FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=event.quantity,
            fill_price=price,
            commission=0.0,
        )
        events.append(fill)
