"""
Cross-Asset Time-Series Momentum (TSMOM) and Cross-Sectional Momentum (XSMOM).
Follows Moskowitz, Ooi, Pedersen (2012) methodology with volatility scaling.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import SignalView
from ramp.signals.base import BaseSignal

class TimeSeriesMomentumSignal(BaseSignal):
    """
    Multi-horizon trend following signal with ex-ante volatility scaling.
    Combines 1M, 3M, and 12M lookback windows.
    """

    def __init__(
        self,
        lookbacks: List[int] = [21, 63, 252],
        weights: Optional[List[float]] = None,
        vol_lookback: int = 63,
        target_vol: float = 0.15,
        clip_zscore: float = 3.0
    ):
        super().__init__(name="tsmom")
        self.lookbacks = lookbacks
        if weights is not None and len(weights) == len(lookbacks):
            self.weights = np.array(weights) / np.sum(weights)
        else:
            self.weights = np.ones(len(lookbacks)) / len(lookbacks)
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.clip_zscore = clip_zscore

    def generate_views(
        self,
        bars_history: pd.DataFrame,
        as_of_date: datetime,
        symbols: List[str]
    ) -> Dict[str, SignalView]:
        views: Dict[str, SignalView] = {}
        df = bars_history[bars_history["timestamp"] <= as_of_date].copy()

        for symbol in symbols:
            sym_df = df[df["symbol"] == symbol].sort_values("timestamp")
            if len(sym_df) < max(self.lookbacks) + 5:

                views[symbol] = SignalView(symbol=symbol, expected_return=0.0, confidence=0.1)
                continue

            closes = sym_df["close"].values
            returns = np.diff(closes) / closes[:-1]

            recent_ret = returns[-self.vol_lookback:]
            vol = float(np.std(recent_ret) * np.sqrt(252))
            vol = max(vol, 0.04)                                                 

            trend_scores = []
            for L in self.lookbacks:
                horizon_ret = (closes[-1] - closes[-L]) / closes[-L]
                expected_horizon_vol = vol * np.sqrt(L / 252.0)
                z = horizon_ret / max(expected_horizon_vol, 1e-4)
                z = np.clip(z, -self.clip_zscore, self.clip_zscore)
                trend_scores.append(z)

            composite_z = float(np.dot(self.weights, trend_scores))

            expected_ret = (self.target_vol / vol) * composite_z * 0.05
            expected_ret = float(np.clip(expected_ret, -0.40, 0.40))

            signs = [np.sign(s) for s in trend_scores]
            agreement = abs(sum(signs)) / len(signs)                                   
            confidence = float(np.clip(0.3 + 0.6 * agreement, 0.1, 0.95))

            views[symbol] = SignalView(
                symbol=symbol,
                expected_return=expected_ret,
                confidence=confidence,
                horizon_bars=21,
                metadata={"composite_z": composite_z, "realized_vol": vol}
            )

        return views
