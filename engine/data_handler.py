"""Data handler: feeds price bars into the backtest one day at a time.

Enforces the no-lookahead rule via a cursor marking 'today'. The strategy can
only request data up to and including the cursor -- never beyond it.

Can also take an injected `prices` DataFrame instead of downloading, which lets
us feed synthetic or sliced data (used for testing and for Phase 5's train/test
splits).
"""
from engine.events import MarketEvent


class PairDataHandler:
    def __init__(self, symbol_a, symbol_b, start="2015-01-01", end=None, prices=None):
        if prices is not None:
            self.prices = prices
        else:
            # Lazy import so injected-data / testing paths don't need yfinance.
            from data.loader import load_pair
            self.prices = load_pair(symbol_a, symbol_b, start=start, end=end)

        self.symbols = [symbol_a, symbol_b]
        self._cursor = -1          # -1 = simulation hasn't started
        self.finished = False

    def update_bars(self, events):
        """Advance the cursor one day and push a MarketEvent (the heartbeat)."""
        self._cursor += 1
        if self._cursor >= len(self.prices):
            self.finished = True
            return
        events.append(MarketEvent(timestamp=self.prices.index[self._cursor]))

    def get_latest(self, symbol, n=1):
        """Last n closes up to the cursor -- never past it (no lookahead)."""
        if self._cursor < 0:
            return []
        start = max(0, self._cursor - n + 1)
        return self.prices[symbol].iloc[start:self._cursor + 1].tolist()

    def current_timestamp(self):
        if self._cursor < 0:
            return None
        return self.prices.index[self._cursor]

    def current_price(self, symbol):
        latest = self.get_latest(symbol, 1)
        return latest[-1] if latest else None
