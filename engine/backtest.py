"""The event loop: drives the simulation and routes every event to its handler.

Structure is two nested loops:
  OUTER (time): advance one bar -> drops a MarketEvent on the queue.
  INNER (drain): pop events and route each by its .type tag. Events created
                 mid-drain (a signal spawns an order spawns a fill) get handled
                 in the same pass, so a full chain resolves within one bar.

The routing itself IS the forward-only chain:
  MARKET -> strategy -> SIGNAL -> portfolio -> ORDER -> execution -> FILL -> portfolio
"""
from collections import deque, defaultdict


class Backtest:
    def __init__(self, data, strategy, portfolio, execution):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events = deque()
        self.counts = defaultdict(int)   # tally of each event type, for inspection

    def run(self):
        while True:
            # OUTER LOOP: advance time by one bar.
            self.data.update_bars(self.events)
            if self.data.finished:
                break

            # INNER LOOP: drain everything the new bar set in motion.
            while self.events:
                event = self.events.popleft()
                self.counts[event.type] += 1

                if event.type == "MARKET":
                    self.strategy.calculate_signals(event, self.data, self.events)
                    self.portfolio.mark_to_market(self.data)
                elif event.type == "SIGNAL":
                    self.portfolio.on_signal(event, self.data, self.events)
                elif event.type == "ORDER":
                    self.execution.on_order(event, self.data, self.events)
                elif event.type == "FILL":
                    self.portfolio.on_fill(event)

        return self.counts
