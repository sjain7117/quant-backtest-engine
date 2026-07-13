"""Cross-sectional momentum: long recent winners, short recent losers.

The opposite hypothesis to pairs mean-reversion: assets that have been going up
tend to KEEP going up (and losers keep losing). Each rebalance we:
  1. score every asset by its trailing return over `lookback` days, SKIPPING the
     most recent `skip` days (classic momentum skips the last month, which tends
     to reverse),
  2. rank them, go LONG the top `n_long` and SHORT the bottom `n_short`,
  3. size the book dollar-neutral and equal-weight.

Rebalancing is CALENDAR-ANCHORED (first trading day of each month), like a real
fund -- so results don't depend on where the data slice happens to start.

No parameter is fitted on data -- lookback/skip are fixed hyperparameters -- so
there is nothing that could leak from training.
"""
from engine.events import SignalEvent
from engine.strategy import Strategy


class CrossSectionalMomentum(Strategy):
    def __init__(self, symbols, lookback=126, skip=21,
                 n_long=3, n_short=3, gross_per_side=50_000):
        self.symbols = list(symbols)
        self.lookback = lookback
        self.skip = skip
        self.n_long = n_long
        self.n_short = n_short
        self.gross_per_side = gross_per_side
        self._last_month = None           # (year, month) of the last rebalance

    def calculate_signals(self, event, data, events):
        need = self.lookback + self.skip
        if len(data.get_latest(self.symbols[0], need)) < need:
            return                        # not enough history yet

        month = (event.timestamp.year, event.timestamp.month)
        if month == self._last_month:
            return                        # rebalance once per calendar month
        self._last_month = month

        scores = {}
        for s in self.symbols:
            hist = data.get_latest(s, need)
            if len(hist) < need:
                return
            past = hist[0]
            recent = hist[len(hist) - 1 - self.skip]
            if past > 0:
                scores[s] = recent / past - 1
        if len(scores) < self.n_long + self.n_short:
            return

        ranked = sorted(scores, key=scores.get, reverse=True)
        longs, shorts = ranked[:self.n_long], ranked[-self.n_short:]

        targets = {s: 0 for s in self.symbols}
        for s in longs:
            targets[s] = int(round((self.gross_per_side / self.n_long) / data.current_price(s)))
        for s in shorts:
            targets[s] = -int(round((self.gross_per_side / self.n_short) / data.current_price(s)))

        for s in self.symbols:
            q = targets[s]
            direction = "LONG" if q > 0 else "SHORT" if q < 0 else "EXIT"
            events.append(SignalEvent(event.timestamp, s, direction, target_quantity=q))
