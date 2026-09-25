from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import SignalView
from ramp.signals.base import BaseSignal

class COTPositioningSignal(BaseSignal):

    def __init__(self, rolling_weeks: int = 52, crowding_threshold_z: float = 2.0):
        super().__init__(name="cot_positioning")
        self.rolling_weeks = rolling_weeks
        self.threshold = crowding_threshold_z

    @staticmethod
    def calculate_positioning_zscore(net_positions: pd.Series, window: int = 52) -> pd.Series:
        rolling_mean = net_positions.rolling(window=window, min_periods=10).mean()
        rolling_std = net_positions.rolling(window=window, min_periods=10).std()
        return (net_positions - rolling_mean) / np.maximum(rolling_std, 1e-4)

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
            if len(sym_df) < 52:
                views[symbol] = SignalView(symbol=symbol, expected_return=0.0, confidence=0.1)
                continue

            if "cot_net_speculative" in sym_df.columns:
                net_series = sym_df["cot_net_speculative"]
                z = float(self.calculate_positioning_zscore(net_series, self.rolling_weeks).iloc[-1])
            else:
                closes = sym_df["close"].values
                ret_52w = (closes[-1] - closes[-min(len(closes), 252)]) / closes[-min(len(closes), 252)]
                z = float(np.clip(ret_52w / 0.15, -3.0, 3.0))

            if z >= self.threshold:
                exp_ret = -0.10
                conf = 0.70
            elif z <= -self.threshold:
                exp_ret = 0.12
                conf = 0.75
            else:
                exp_ret = 0.02 * (-z / self.threshold)
                conf = 0.35

            views[symbol] = SignalView(
                symbol=symbol,
                expected_return=float(exp_ret),
                confidence=float(conf),
                horizon_bars=63,
                metadata={"positioning_zscore": round(z, 2)}
            )

        return views
