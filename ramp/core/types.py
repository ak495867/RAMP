"""
Core type definitions, enums, and data models for RAMP.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    TWAP = "TWAP"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class Bar:
    """Standardized Point-in-Time OHLCV Bar."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    vwap: Optional[float] = None

    def __post_init__(self):
        if self.high < self.low:
            raise ValueError(f"High {self.high} cannot be less than low {self.low} for {self.symbol}")
        if self.volume < 0:
            raise ValueError(f"Volume {self.volume} cannot be negative for {self.symbol}")


@dataclass
class Order:
    """Order specification dispatched by portfolio optimizer."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.PENDING


@dataclass
class Fill:
    """Execution fill returned by execution simulator or broker."""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float
    timestamp: datetime


@dataclass
class Position:
    """Tracks current holdings in an asset."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0.0:
            return 0.0
        return (self.current_price - self.avg_price) * self.quantity


@dataclass
class RegimeState:
    """Regime classification output at timestamp t."""
    timestamp: datetime
    regime_id: int
    regime_name: str
    probabilities: Dict[int, float]
    is_transition: bool = False
    entropy: float = 0.0


@dataclass
class SignalView:
    """Alpha signal output with expected return and view uncertainty."""
    symbol: str
    expected_return: float
    confidence: float  # 0.0 to 1.0
    horizon_bars: int = 21
    metadata: Dict = field(default_factory=dict)
