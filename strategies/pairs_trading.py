"""Statistical-arbitrage pairs strategy on a cointegrated pair.

spread_t = price_A - beta * price_B     (beta is STATIC, fixed from training)

We z-score the spread against its RECENT past only (a rolling window), so there
is no lookahead. Trading rules:
  z > +entry  -> spread is rich  -> SHORT the spread (short A, long beta*B)
  z < -entry  -> spread is cheap -> LONG  the spread (long A, short beta*B)
  |z| < exit  -> revert achieved -> CLOSE both legs
  |z| > stop  -> spread ran away -> CLOSE both legs (cut the loss)
"""
import numpy as np
from engine.events import SignalEvent
from engine.strategy import Strategy


class PairsTradingStrategy(Strategy):
    def __init__(self, sym_a, sym_b, beta, lookback=30,
                 entry_z=2.0, exit_z=0.5, stop_z=3.5, base_units=100):
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.beta = beta
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.base_units = base_units
        self.state = "FLAT"          # FLAT | LONG_SPREAD | SHORT_SPREAD

    def _emit_targets(self, ts, target_a, target_b, events):
        def direction(q):
            return "LONG" if q > 0 else "SHORT" if q < 0 else "EXIT"
        events.append(SignalEvent(ts, self.sym_a, direction(target_a), target_quantity=target_a))
        events.append(SignalEvent(ts, self.sym_b, direction(target_b), target_quantity=target_b))

    def calculate_signals(self, event, data, events):
        a = data.get_latest(self.sym_a, self.lookback)
        b = data.get_latest(self.sym_b, self.lookback)
        if len(a) < self.lookback or len(b) < self.lookback:
            return

        spread = np.array(a) - self.beta * np.array(b)
        std = spread.std(ddof=1)
        if std == 0:
            return
        z = (spread[-1] - spread.mean()) / std           # rolling z of the latest bar

        units = self.base_units
        hedged = int(round(units * self.beta))           # B-leg size from hedge ratio

        if self.state == "FLAT":
            if z > self.entry_z:                          # short the spread
                self._emit_targets(event.timestamp, -units, +hedged, events)
                self.state = "SHORT_SPREAD"
            elif z < -self.entry_z:                       # long the spread
                self._emit_targets(event.timestamp, +units, -hedged, events)
                self.state = "LONG_SPREAD"

        elif self.state == "SHORT_SPREAD":
            if z < self.exit_z or z > self.stop_z:        # reverted, or stopped out
                self._emit_targets(event.timestamp, 0, 0, events)
                self.state = "FLAT"

        elif self.state == "LONG_SPREAD":
            if z > -self.exit_z or z < -self.stop_z:      # reverted, or stopped out
                self._emit_targets(event.timestamp, 0, 0, events)
                self.state = "FLAT"
