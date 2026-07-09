"""Base Strategy class. Real strategies subclass this and implement the logic."""


class Strategy:
    def calculate_signals(self, event, data, events):
        """Look at data up to the current bar and optionally append SignalEvents.

        event  : the MarketEvent that just fired
        data   : the data handler (only exposes prices up to the cursor)
        events : the shared queue -- append SignalEvent(s) here to act
        """
        raise NotImplementedError("Subclasses must implement calculate_signals")
