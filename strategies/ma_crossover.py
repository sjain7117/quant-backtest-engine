"""Moving-average crossover: a simple trend-following rule.

Track a SHORT and a LONG moving average of the price:
  short crosses ABOVE long  -> uptrend beginning   -> go LONG
  short crosses BELOW long  -> uptrend ending       -> EXIT

This is NOT a good strategy -- trend-following a single stock is mostly noise.
We use it purely to stress-test the engine's ability to enter and exit many
times over the backtest, which buy-and-hold (one trade) never exercised.
"""
from engine.events import SignalEvent
from engine.strategy import Strategy


class MACrossover(Strategy):
    def __init__(self, symbol="KO", short_window=20, long_window=100):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.in_market = False   # are we currently holding a long position?

    def calculate_signals(self, event, data, events):
        # Need at least `long_window` bars of history before averages are valid.
        prices = data.get_latest(self.symbol, self.long_window)
        if len(prices) < self.long_window:
            return

        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices) / self.long_window

        # Entry: short above long, and we're currently flat.
        if short_ma > long_ma and not self.in_market:
            events.append(SignalEvent(event.timestamp, self.symbol, "LONG"))
            self.in_market = True
        # Exit: short falls below long, and we're currently long.
        elif short_ma < long_ma and self.in_market:
            events.append(SignalEvent(event.timestamp, self.symbol, "EXIT"))
            self.in_market = False
