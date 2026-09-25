from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ramp.core.types import Order, OrderSide

@dataclass
class ComplianceResult:
    is_compliant: bool
    violations: List[str]
    clamped_orders: List[Order]
    kill_switch_active: bool = False

class PreTradeComplianceEngine:

    def __init__(
        self,
        max_adv_participation: float = 0.05,
        max_single_weight: float = 0.15,
        max_gross_leverage: float = 1.50,
        max_net_leverage: float = 1.00,
        drawdown_kill_switch_pct: float = -0.05,
        asset_class_caps: Optional[Dict[str, float]] = None
    ):
        self.max_adv_participation = max_adv_participation
        self.max_single_weight = max_single_weight
        self.max_gross_leverage = max_gross_leverage
        self.max_net_leverage = max_net_leverage
        self.drawdown_kill_switch_pct = drawdown_kill_switch_pct
        self.asset_class_caps = asset_class_caps or {
            "equity": 0.55,
            "crypto": 0.15,
            "commodity": 0.35,
            "fixed_income": 0.70
        }
        self.kill_switch_triggered = False

    def check_circuit_breakers(self, current_drawdown: float) -> bool:
        if current_drawdown <= self.drawdown_kill_switch_pct:
            self.kill_switch_triggered = True
            return True
        return False

    def validate_orders(
        self,
        orders: List[Order],
        current_nav: float,
        adv_map: Dict[str, float],
        price_map: Dict[str, float],
        asset_class_map: Optional[Dict[str, str]] = None,
        current_drawdown: float = 0.0
    ) -> ComplianceResult:
        violations = []
        clamped_orders = []

        if self.check_circuit_breakers(current_drawdown):
            violations.append(
                f"KILL SWITCH ACTIVE: Current drawdown {current_drawdown*100:.2f}% breached stop {self.drawdown_kill_switch_pct*100:.2f}%"
            )
            return ComplianceResult(
                is_compliant=False,
                violations=violations,
                clamped_orders=[],
                kill_switch_active=True
            )

        asset_class_dollar_totals: Dict[str, float] = {}
        total_gross_dollar = 0.0
        total_net_dollar = 0.0

        for order in orders:
            sym = order.symbol
            price = price_map.get(sym, 100.0)
            adv = adv_map.get(sym, 1000000.0)

            max_allowed_qty = adv * self.max_adv_participation
            if order.quantity > max_allowed_qty:
                violations.append(
                    f"ADV participation violation for {sym}: requested {order.quantity:.0f}, max allowed {max_allowed_qty:.0f}"
                )
                clamped_qty = max_allowed_qty
            else:
                clamped_qty = order.quantity

            order_dollar = clamped_qty * price
            max_single_dollar = current_nav * self.max_single_weight
            if order_dollar > max_single_dollar:
                violations.append(
                    f"Concentration violation for {sym}: {order_dollar/current_nav*100:.1f}% exceeds max {self.max_single_weight*100:.1f}%"
                )
                clamped_qty = max_single_dollar / max(price, 1e-4)

            total_gross_dollar += clamped_qty * price
            total_net_dollar += (clamped_qty * price) if order.side == OrderSide.BUY else -(clamped_qty * price)

            if asset_class_map and sym in asset_class_map:
                ac = asset_class_map[sym]
                asset_class_dollar_totals[ac] = asset_class_dollar_totals.get(ac, 0.0) + (clamped_qty * price)

            clamped_orders.append(
                Order(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=float(clamped_qty),
                    order_type=order.order_type,
                    limit_price=order.limit_price,
                    created_at=order.created_at,
                    status=order.status
                )
            )

        gross_lev = total_gross_dollar / max(current_nav, 1.0)
        if gross_lev > self.max_gross_leverage:
            violations.append(
                f"Gross leverage violation: {gross_lev:.2f}x exceeds limit {self.max_gross_leverage:.2f}x"
            )

        net_lev = abs(total_net_dollar) / max(current_nav, 1.0)
        if net_lev > self.max_net_leverage:
            violations.append(
                f"Net leverage violation: {net_lev:.2f}x exceeds limit {self.max_net_leverage:.2f}x"
            )

        if asset_class_map:
            for ac, dollar_val in asset_class_dollar_totals.items():
                cap = self.asset_class_caps.get(ac, 1.0)
                if (dollar_val / max(current_nav, 1.0)) > cap:
                    violations.append(
                        f"Asset class cap violation for {ac}: {dollar_val/current_nav*100:.1f}% exceeds cap {cap*100:.1f}%"
                    )

        return ComplianceResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            clamped_orders=clamped_orders,
            kill_switch_active=False
        )
