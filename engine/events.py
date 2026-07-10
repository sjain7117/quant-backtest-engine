"""Event classes: the messages that flow one-way through the backtest."""
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Optional


@dataclass
class MarketEvent:
    type: ClassVar[str] = "MARKET"
    timestamp: datetime


@dataclass
class SignalEvent:
    type: ClassVar[str] = "SIGNAL"
    timestamp: datetime
    symbol: str
    direction: str
    strength: float = 1.0
    target_quantity: Optional[int] = None   # explicit target position; pairs legs use this


@dataclass
class OrderEvent:
    type: ClassVar[str] = "ORDER"
    timestamp: datetime
    symbol: str
    quantity: int
    order_type: str = "MKT"


@dataclass
class FillEvent:
    type: ClassVar[str] = "FILL"
    timestamp: datetime
    symbol: str
    quantity: int
    fill_price: float
    commission: float = 0.0
