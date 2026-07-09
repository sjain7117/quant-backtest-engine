"""Event classes: the messages that flow one-way through the backtest.

The chain is:
    MarketEvent  -> a new day's price is available
    SignalEvent  -> the strategy decided it wants a position
    OrderEvent   -> that decision became a concrete order
    FillEvent    -> the simulated broker executed it; the portfolio updates

Information only ever flows FORWARD along this chain. That is the whole point:
the strategy is handed data one bar at a time, so it physically cannot peek at
future prices. That single guarantee is what makes the backtest trustworthy.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar


@dataclass
class MarketEvent:
    """Fired when a new bar (one day's prices) becomes available.

    Carries only the timestamp; it's a 'wake up, there's new data' signal that
    tells the strategy to look at the market as of this date.
    """
    type: ClassVar[str] = "MARKET"
    timestamp: datetime


@dataclass
class SignalEvent:
    """The strategy's decision about a single symbol, based on data up to now.

    direction is one of:
        "LONG"  -> want a position that profits if the price rises
        "SHORT" -> want a position that profits if the price falls
        "EXIT"  -> want to close any existing position (go flat)

    strength (0..1) is an optional confidence/sizing hint. We'll use it in
    Phase 4 to feed position sizing (the Kelly criterion). Default 1.0 = full.
    """
    type: ClassVar[str] = "SIGNAL"
    timestamp: datetime
    symbol: str
    direction: str
    strength: float = 1.0


@dataclass
class OrderEvent:
    """A concrete instruction to trade, produced from a SignalEvent.

    quantity is SIGNED: +100 means buy 100 shares, -100 means sell 100.
    order_type "MKT" = a market order (execute immediately at the current
    price, as opposed to waiting for a specific price).
    """
    type: ClassVar[str] = "ORDER"
    timestamp: datetime
    symbol: str
    quantity: int
    order_type: str = "MKT"


@dataclass
class FillEvent:
    """A record that an order actually executed ('filled').

    quantity is SIGNED, matching the order.
    fill_price is the price actually achieved (later this will include
    slippage: the small adverse price move between deciding and executing).
    commission is the fee paid to trade (0 for now; added in Phase 3).
    """
    type: ClassVar[str] = "FILL"
    timestamp: datetime
    symbol: str
    quantity: int
    fill_price: float
    commission: float = 0.0
