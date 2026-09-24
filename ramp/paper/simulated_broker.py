"""
Paper broker interfaces and drift reconciler.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ramp.core.types import Fill, Order, Position


class BaseBroker(ABC):
    """Abstract broker interface for live and paper trading."""

    @abstractmethod
    def get_nav(self) -> float:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> Fill:
        pass


class SimulatedPaperBroker(BaseBroker):
    """
    Local paper trading broker tracking actual fills, account cash, and portfolio drift.
    """

    def __init__(self, initial_cash: float = 1000000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.fills: List[Fill] = []

    def get_nav(self) -> float:
        invested = sum(pos.market_value for pos in self.positions.values())
        return self.cash + invested

    def get_positions(self) -> Dict[str, Position]:
        return self.positions

    def submit_order(self, order: Order) -> Fill:
        self.orders.append(order)
        # Mock execution at nominal price
        fill_price = order.limit_price or 100.0
        fill = Fill(
            fill_id=f"paper_fill_{len(self.fills) + 1}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=1.0,
            slippage=0.02 * order.quantity,
            timestamp=order.created_at
        )
        self.fills.append(fill)

        cost = fill.quantity * fill.price
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(symbol=order.symbol)

        pos = self.positions[order.symbol]
        if order.side.value == "BUY":
            self.cash -= (cost + fill.commission)
            pos.quantity += fill.quantity
            pos.current_price = fill.price
        else:
            self.cash += (cost - fill.commission)
            pos.quantity -= fill.quantity
            pos.current_price = fill.price

        return fill
