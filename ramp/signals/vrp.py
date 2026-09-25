"""
Volatility Risk Premium (VRP) Signal.
Exploits the structural spread between implied volatility (options market) and realized volatility.
"""

from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import SignalView
from ramp.signals.base import BaseSignal

class VolatilityRiskPremiumSignal(BaseSignal):
    """
    Computes VRP = Implied Volatility - Realized Volatility.
    In calm regimes, positive VRP indicates premium collection opportunity (long equity/risk beta).
    In crisis regimes, negative or inverted VRP flags volatility explosion (risk-off).
    """

    def __init__(self, rv_lookback: int = 21, target_equity_sym: str = "SPY"):
        super().__init__(name="vrp")
        self.rv_lookback = rv_lookback
        self.target_equity_sym = target_equity_sym

    def generate_views(
        self,
        bars_history: pd.DataFrame,
        as_of_date: datetime,
        symbols: List[str]
    ) -> Dict[str, SignalView]:
        views: Dict[str, SignalView] = {}
        df = bars_history[bars_history["timestamp"] <= as_of_date]

        eq_df = df[df["symbol"] == self.target_equity_sym].sort_values("timestamp")
        if len(eq_df) < self.rv_lookback + 5:
            for s in symbols:
                views[s] = SignalView(symbol=s, expected_return=0.0, confidence=0.1)
            return views

        closes = eq_df["close"].values
        daily_ret = np.diff(closes) / closes[:-1]
        rv = float(np.std(daily_ret[-self.rv_lookback:]) * np.sqrt(252))

        vix_df = df[df["symbol"] == "^VIX"].sort_values("timestamp")
        if not vix_df.empty and len(vix_df) > 0:
            iv = float(vix_df["close"].iloc[-1]) / 100.0
        else:

            iv = rv * 1.15

        vrp_spread = iv - rv

        for symbol in symbols:

            if vrp_spread > 0.02:

                exp_ret = 0.08 if symbol in ["SPY", "QQQ", "BTC-USD"] else -0.02
                conf = 0.65
            elif vrp_spread < -0.02:

                exp_ret = -0.15 if symbol in ["SPY", "QQQ", "IWM", "BTC-USD"] else 0.10
                conf = 0.85
            else:
                exp_ret = 0.0
                conf = 0.30

            views[symbol] = SignalView(
                symbol=symbol,
                expected_return=float(exp_ret),
                confidence=conf,
                horizon_bars=21,
                metadata={"vrp_spread": vrp_spread, "iv": iv, "rv": rv}
            )

        return views
