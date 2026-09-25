"""
Realistic Transaction Cost and Non-Linear Market Impact Simulator.
Implements the Kyle / Almgren-Chriss square-root price impact model,
bid-ask spread crossing, broker commissions, and liquidity caps.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from ramp.core.types import AssetClass, OrderSide

class ExecutionCostModel:
    """
    Computes realistic execution prices and fees.

    Price Impact formulation:
      Impact = Y * sigma_daily * sqrt(Q / ADV)
      P_exec = P_mid * (1 + (side_sign) * (half_spread + Impact + delay_slippage))
    """

    def __init__(
        self,
        impact_coefficient_y: float = 0.15,
        default_half_spread_bps: float = 2.0,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        max_adv_participation: float = 0.05,
        delay_slippage_bps: float = 1.0
    ):
        self.y = impact_coefficient_y
        self.half_spread = default_half_spread_bps / 10000.0
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_adv_participation = max_adv_participation
        self.delay_slippage = delay_slippage_bps / 10000.0

    def calculate_fill(
        self,
        side: OrderSide,
        requested_qty: float,
        mid_price: float,
        daily_volume: float,
        daily_volatility: float,
        asset_class: AssetClass = AssetClass.EQUITY
    ) -> Tuple[float, float, float, float]:
        """
        Calculates execution details.

        Returns:
          (executed_qty, fill_price, total_commission, total_slippage_cost)
        """
        if requested_qty <= 0 or mid_price <= 0:
            return 0.0, mid_price, 0.0, 0.0

        max_executable = max(daily_volume * self.max_adv_participation, 1.0)
        executed_qty = min(requested_qty, max_executable)

        participation_rate = executed_qty / max(daily_volume, 1.0)
        market_impact_pct = self.y * daily_volatility * np.sqrt(participation_rate)

        total_friction_pct = self.half_spread + market_impact_pct + self.delay_slippage

        sign = 1.0 if side == OrderSide.BUY else -1.0
        fill_price = mid_price * (1.0 + sign * total_friction_pct)

        slippage_cost = abs(fill_price - mid_price) * executed_qty

        if asset_class == AssetClass.CRYPTO:
            commission = max((mid_price * executed_qty) * 0.0010, 0.50)                    
        else:
            commission = max(executed_qty * self.commission_per_share, self.min_commission)

        return executed_qty, float(fill_price), float(commission), float(slippage_cost)
