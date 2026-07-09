"""Buy-and-hold: go LONG on the first bar, never trade again.

This is a VALIDATION strategy, not a real one. We know exactly what it should
earn -- the asset's own price appreciation -- so if the engine reports a
different return, there's a bug in the accounting. Better to catch it here on a
trivial case than let it silently corrupt the pairs results later.
"""
from engine.events import SignalEvent
from engine.strategy import Strategy


class BuyAndHold(Strategy):
    def __init__(self, symbol="KO"):
        self.symbol = symbol
        self.invested = False

    def calculate_signals(self, event, data, events):
        # Fire exactly once, on the very first bar we see.
        if not self.invested:
            events.append(SignalEvent(event.timestamp, self.symbol, "LONG"))
            self.invested = True
