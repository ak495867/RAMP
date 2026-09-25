"""
Portfolio Ledger and Accounting Engine.
Maintains exact point-in-time cash, margin, positions, and realized/unrealized P&L.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from ramp.core.types import Fill, OrderSide, Position

class PortfolioLedger:
    """
    Maintains the state of capital, open positions, borrowing costs, and transaction history.
    """

    def __init__(self, initial_capital: float = 1000000.0, risk_free_rate_annual: float = 0.045):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.risk_free_rate = risk_free_rate_annual
        self.positions: Dict[str, Position] = {}
        self.fills_history: List[Fill] = []
        self.history_records: List[Dict] = []

    def record_fill(self, fill: Fill) -> None:
        """Updates cash and position based on execution fill."""
        self.fills_history.append(fill)
        sym = fill.symbol

        if sym not in self.positions:
            self.positions[sym] = Position(symbol=sym)

        pos = self.positions[sym]
        cost = fill.quantity * fill.price

        if fill.side == OrderSide.BUY:
            self.cash -= (cost + fill.commission)

            new_qty = pos.quantity + fill.quantity
            if new_qty > 0:
                pos.avg_price = (pos.quantity * pos.avg_price + cost) / new_qty
            pos.quantity = new_qty
        else:                  
            self.cash += (cost - fill.commission)

            realized = (fill.price - pos.avg_price) * fill.quantity
            pos.realized_pnl += realized
            pos.quantity -= fill.quantity

        pos.current_price = fill.price

    def accrue_daily_interest(self, dt: datetime) -> None:
        """Accrues interest on unallocated cash balance (252-day basis)."""
        daily_rf = self.risk_free_rate / 252.0
        if self.cash > 0:
            interest = self.cash * daily_rf
            self.cash += interest

    def mark_to_market(self, dt: datetime, prices: Dict[str, float]) -> Dict:
        """
        Marks all active holdings to market and records daily snapshot.
        """
        invested_equity = 0.0
        gross_exposure = 0.0

        for sym, price in prices.items():
            if sym in self.positions:
                pos = self.positions[sym]
                pos.current_price = price
                val = pos.market_value
                invested_equity += val
                gross_exposure += abs(val)

        total_nav = self.cash + invested_equity
        leverage = gross_exposure / max(total_nav, 1.0)

        record = {
            "timestamp": dt,
            "cash": self.cash,
            "invested_equity": invested_equity,
            "total_nav": total_nav,
            "gross_exposure": gross_exposure,
            "leverage": leverage,
        }
        self.history_records.append(record)
        return record

    def get_history_df(self) -> pd.DataFrame:
        """Returns time-series dataframe of portfolio performance."""
        if not self.history_records:
            return pd.DataFrame()
        df = pd.DataFrame(self.history_records)
        df["daily_return"] = df["total_nav"].pct_change().fillna(0.0)
        return df
