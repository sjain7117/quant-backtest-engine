"""Throwaway strategy to prove the event chain routes end to end.

Goes LONG on the 5th bar, EXITs on the 10th -- just enough to watch a full
SIGNAL -> ORDER -> FILL cycle flow through twice. Deleted in Phase 2.
"""
from engine.events import SignalEvent
from engine.strategy import Strategy


class SmokeTestStrategy(Strategy):
    def __init__(self, symbol="KO"):
        self.symbol = symbol
        self.bar = 0

    def calculate_signals(self, event, data, events):
        self.bar += 1
        if self.bar == 5:
            events.append(SignalEvent(event.timestamp, self.symbol, "LONG"))
        elif self.bar == 10:
            events.append(SignalEvent(event.timestamp, self.symbol, "EXIT"))
