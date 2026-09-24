"""
Mean Reversion & Value Spread Signal.
Detects temporary liquidity dislocations and statistical overshoot.
"""

from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd
from ramp.core.types import SignalView
from ramp.signals.base import BaseSignal


class MeanReversionSignal(BaseSignal):
    """
    Computes rolling z-score of price relative to exponential moving average.
    Contrarian expected return: expects reversion towards the mean.
    """

    def __init__(self, lookback: int = 20, entry_zscore: float = 2.0, max_expected_ret: float = 0.25):
        super().__init__(name="mean_reversion")
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.max_expected_ret = max_expected_ret

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
            if len(sym_df) < self.lookback + 5:
                views[symbol] = SignalView(symbol=symbol, expected_return=0.0, confidence=0.1)
                continue

            closes = sym_df["close"].values
            window = closes[-self.lookback:]
            mean_p = float(np.mean(window))
            std_p = float(np.std(window))

            if std_p <= 1e-6:
                views[symbol] = SignalView(symbol=symbol, expected_return=0.0, confidence=0.1)
                continue

            current_p = closes[-1]
            z_score = (current_p - mean_p) / std_p

            # Reversion expected return is negatively proportional to z-score
            expected_ret = float(-z_score * 0.05)
            expected_ret = np.clip(expected_ret, -self.max_expected_ret, self.max_expected_ret)

            # High confidence only when dislocation is extreme (|z| >= entry_zscore)
            dislocation_intensity = min(abs(z_score) / self.entry_zscore, 1.0)
            confidence = float(0.2 + 0.6 * dislocation_intensity)

            views[symbol] = SignalView(
                symbol=symbol,
                expected_return=expected_ret,
                confidence=confidence,
                horizon_bars=self.lookback,
                metadata={"z_score": z_score}
            )

        return views
