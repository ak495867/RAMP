from typing import Dict, List, Optional, Tuple
import numpy as np
from ramp.core.types import Order, OrderSide, OrderType, Position

class PortfolioDriftReconciler:

    def __init__(self, drift_tolerance_pct: float = 0.01, min_trade_dollar: float = 100.0):
        self.drift_tolerance = drift_tolerance_pct
        self.min_trade_dollar = min_trade_dollar

    def calculate_rebalance_orders(
        self,
        current_nav: float,
        current_positions: Dict[str, Position],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> Tuple[List[Order], Dict[str, float], bool]:
        orders: List[Order] = []
        drift_map: Dict[str, float] = {}
        all_symbols = sorted(set(list(current_positions.keys()) + list(target_weights.keys())))
        rebalance_required = False

        for sym in all_symbols:
            price = current_prices.get(sym, 100.0)
            curr_pos = current_positions.get(sym, Position(symbol=sym))
            curr_qty = curr_pos.quantity
            curr_dollar = curr_qty * price
            curr_weight = curr_dollar / max(current_nav, 1.0)

            t_weight = target_weights.get(sym, 0.0)
            drift = abs(curr_weight - t_weight)
            drift_map[sym] = round(float(drift), 4)

            if drift > self.drift_tolerance:
                rebalance_required = True
                target_dollar = current_nav * t_weight
                delta_dollar = target_dollar - curr_dollar

                if abs(delta_dollar) >= self.min_trade_dollar:
                    shares_to_trade = abs(delta_dollar) / max(price, 1e-4)
                    side = OrderSide.BUY if delta_dollar > 0 else OrderSide.SELL
                    orders.append(
                        Order(
                            order_id=f"reconcile_{sym}",
                            symbol=sym,
                            side=side,
                            quantity=float(round(shares_to_trade, 4)),
                            order_type=OrderType.MARKET
                        )
                    )

        return orders, drift_map, rebalance_required
