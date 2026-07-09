"""Data handler: feeds price bars into the backtest one day at a time.

The key job here is enforcing the no-lookahead rule. The handler holds the full
price history but keeps a CURSOR marking 'today' in the simulation. The strategy
can only ever request data up to and including the cursor -- never beyond it.
That makes peeking at future prices impossible, not just discouraged.
"""
from collections import deque
from data.loader import load_pair
from engine.events import MarketEvent


class PairDataHandler:
    def __init__(self, symbol_a, symbol_b, start="2015-01-01", end=None):
        # Full aligned price history for the two symbols (dates both traded).
        self.prices = load_pair(symbol_a, symbol_b, start=start, end=end)
        self.symbols = [symbol_a, symbol_b]

        # The cursor: index of 'today'. -1 means the sim hasn't started yet.
        self._cursor = -1

        # Set True once we've walked off the end of the data.
        self.finished = False

    def update_bars(self, events):
        """Advance the cursor one day and push a MarketEvent onto the queue.

        Called once per iteration of the main loop -- it's the heartbeat that
        drives the whole simulation forward through time.
        """
        self._cursor += 1
        if self._cursor >= len(self.prices):
            self.finished = True
            return
        timestamp = self.prices.index[self._cursor]
        events.append(MarketEvent(timestamp=timestamp))

    def get_latest(self, symbol, n=1):
        """Return the last n closing prices for `symbol`, up to the cursor.

        This is the enforcement point: the slice ends at self._cursor + 1, so
        the strategy can NEVER see prices dated after 'today'. Returns a plain
        list of floats, oldest first.
        """
        if self._cursor < 0:
            return []
        start = max(0, self._cursor - n + 1)
        window = self.prices[symbol].iloc[start:self._cursor + 1]
        return window.tolist()

    def current_timestamp(self):
        """The date of the current bar (today in the simulation)."""
        if self._cursor < 0:
            return None
        return self.prices.index[self._cursor]

    def current_price(self, symbol):
        """Today's closing price for `symbol` -- convenience for get_latest(...,1)."""
        latest = self.get_latest(symbol, 1)
        return latest[-1] if latest else None
