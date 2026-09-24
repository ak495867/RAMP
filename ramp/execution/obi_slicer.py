from dataclasses import dataclass
from typing import Dict, List, Optional
from ramp.core.types import OrderSide


@dataclass
class MicrostructureDecision:
    action: str
    target_price_offset_bps: float
    urgency_multiplier: float
    order_book_imbalance: float


class OrderBookImbalanceSlicer:

    def __init__(self, strong_imbalance_threshold: float = 0.30):
        self.threshold = strong_imbalance_threshold

    @staticmethod
    def calculate_obi(bid_volume: float, ask_volume: float) -> float:
        total = bid_volume + ask_volume
        if total <= 0:
            return 0.0
        return float((bid_volume - ask_volume) / total)

    def evaluate_microstructure_action(
        self,
        side: OrderSide,
        bid_volume: float,
        ask_volume: float,
        bid_price: float,
        ask_price: float
    ) -> MicrostructureDecision:
        obi = self.calculate_obi(bid_volume, ask_volume)
        spread_bps = ((ask_price - bid_price) / max(bid_price, 1e-4)) * 10000.0

        if side == OrderSide.BUY:
            if obi >= self.threshold:
                action = "AGGRESSIVE_BID_CROSS"
                offset_bps = 0.25 * spread_bps
                urgency = 1.35
            elif obi <= -self.threshold:
                action = "PASSIVE_WAIT_PULLBACK"
                offset_bps = -0.50 * spread_bps
                urgency = 0.65
            else:
                action = "NEUTRAL_MID_PEG"
                offset_bps = 0.0
                urgency = 1.00
        else:
            if obi <= -self.threshold:
                action = "AGGRESSIVE_ASK_CROSS"
                offset_bps = -0.25 * spread_bps
                urgency = 1.35
            elif obi >= self.threshold:
                action = "PASSIVE_WAIT_BOUNCE"
                offset_bps = 0.50 * spread_bps
                urgency = 0.65
            else:
                action = "NEUTRAL_MID_PEG"
                offset_bps = 0.0
                urgency = 1.00

        return MicrostructureDecision(
            action=action,
            target_price_offset_bps=round(float(offset_bps), 2),
            urgency_multiplier=round(float(urgency), 2),
            order_book_imbalance=round(float(obi), 4)
        )
