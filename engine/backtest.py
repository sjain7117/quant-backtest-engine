"""Event loop with next-bar execution and end-of-bar mark-to-market.

Per bar:
  1. a new MarketEvent fires
  2. orders queued on the PREVIOUS bar fill at THIS bar's price
  3. the strategy reacts to this bar (queuing orders for the NEXT bar)
  4. once the queue is drained, we snapshot equity at today's close
"""
from collections import deque, defaultdict


class Backtest:
    def __init__(self, data, strategy, portfolio, execution):
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.events = deque()
        self.counts = defaultdict(int)

    def run(self):
        while True:
            self.data.update_bars(self.events)        # OUTER: advance one bar
            if self.data.finished:
                break

            while self.events:                        # INNER: drain the queue
                event = self.events.popleft()
                self.counts[event.type] += 1

                if event.type == "MARKET":
                    # fill last bar's orders at THIS bar's price, then let the
                    # strategy react (which queues orders for the NEXT bar)
                    self.execution.execute_pending(self.data, self.events)
                    self.strategy.calculate_signals(event, self.data, self.events)
                elif event.type == "SIGNAL":
                    self.portfolio.on_signal(event, self.data, self.events)
                elif event.type == "ORDER":
                    self.execution.on_order(event, self.data, self.events)
                elif event.type == "FILL":
                    self.portfolio.on_fill(event)

            # snapshot equity once everything for this bar has settled
            self.portfolio.mark_to_market(self.data)

        return self.counts
