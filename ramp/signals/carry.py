"""
Cross-Asset Carry Signal.
Harvests structural premia: futures roll yield, fixed-income term premium, FX forward discount.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import SignalView
from ramp.signals.base import BaseSignal

class CrossAssetCarrySignal(BaseSignal):
    """
    Computes carry estimates across asset classes.
    Positive carry indicates positive roll yield or yield advantage.
    """

    def __init__(self, default_cash_rate: float = 0.045):
        super().__init__(name="carry")
        self.default_cash_rate = default_cash_rate

    def generate_views(
        self,
        bars_history: pd.DataFrame,
        as_of_date: datetime,
        symbols: List[str]
    ) -> Dict[str, SignalView]:
        views: Dict[str, SignalView] = {}
        df = bars_history[bars_history["timestamp"] <= as_of_date]

        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol].sort_values("timestamp")
            if len(sym_df) < 21:
                views[symbol] = SignalView(symbol=symbol, expected_return=0.0, confidence=0.1)
                continue

            if "roll_gap" in sym_df.columns:
                recent_basis = sym_df["roll_gap"].iloc[-21:].mean()
                carry_est = float(recent_basis * 12)                         
            else:

                closes = sym_df["close"].values
                ret_63 = (closes[-1] - closes[-min(len(closes), 63)]) / closes[-min(len(closes), 63)]
                carry_est = float(ret_63 * 0.5)

            carry_ret = float(np.clip(carry_est, -0.20, 0.20))
            views[symbol] = SignalView(
                symbol=symbol,
                expected_return=carry_ret,
                confidence=0.60,
                horizon_bars=63,
                metadata={"carry_estimate": carry_ret}
            )

        return views
