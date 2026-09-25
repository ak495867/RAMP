from datetime import datetime
from typing import Dict, List, Optional
import requests
from ramp.core.types import Fill, Order, OrderSide, OrderType, Position
from ramp.paper.simulated_broker import BaseBroker

class AlpacaBrokerGateway(BaseBroker):

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        is_paper: bool = True
    ):
        self.api_key = api_key or "MOCK_ALPACA_KEY"
        self.secret_key = secret_key or "MOCK_ALPACA_SECRET"
        self.base_url = (
            "https://paper-api.alpaca.markets/v2" if is_paper
            else "https://api.alpaca.markets/v2"
        )
        self.mock_mode = (api_key is None or api_key == "MOCK_ALPACA_KEY")
        self.mock_cash = 1000000.0
        self.mock_positions: Dict[str, Position] = {}
        self.fills_log: List[Fill] = []

    def get_nav(self) -> float:
        if self.mock_mode:
            invested = sum(p.market_value for p in self.mock_positions.values())
            return float(self.mock_cash + invested)
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        res = requests.get(f"{self.base_url}/account", headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return float(data.get("equity", 0.0))
        return 0.0

    def get_positions(self) -> Dict[str, Position]:
        if self.mock_mode:
            return self.mock_positions
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        res = requests.get(f"{self.base_url}/positions", headers=headers, timeout=10)
        positions = {}
        if res.status_code == 200:
            for item in res.json():
                sym = item["symbol"]
                positions[sym] = Position(
                    symbol=sym,
                    quantity=float(item["qty"]),
                    avg_price=float(item["avg_entry_price"]),
                    current_price=float(item["current_price"])
                )
        return positions

    def submit_order(self, order: Order) -> Fill:
        fill_price = order.limit_price or 100.0
        fill = Fill(
            fill_id=f"alpaca_fill_{len(self.fills_log) + 1}",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=0.0,
            slippage=0.0001 * fill_price * order.quantity,
            timestamp=datetime.utcnow()
        )
        self.fills_log.append(fill)

        cost = fill.quantity * fill.price
        if order.symbol not in self.mock_positions:
            self.mock_positions[order.symbol] = Position(symbol=order.symbol)

        pos = self.mock_positions[order.symbol]
        if order.side == OrderSide.BUY:
            self.mock_cash -= cost
            pos.quantity += fill.quantity
            pos.current_price = fill.price
        else:
            self.mock_cash += cost
            pos.quantity -= fill.quantity
            pos.current_price = fill.price

        return fill
