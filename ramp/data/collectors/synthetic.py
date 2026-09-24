"""
Synthetic multi-asset data generator driven by a ground-truth Markov Switching process.
Essential for offline validation, unit testing, and verifying regime detector accuracy.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from ramp.data.collectors.base import BaseCollector


class SyntheticRegimeDataGenerator(BaseCollector):
    """
    Generates synthetic multi-asset OHLCV data conditioned on a known ground-truth Markov chain.
    
    Regimes:
      0: Low-Vol Bull (High positive equity drift, low vol, positive carry, low correlations)
      1: High-Vol Bear (Negative equity drift, high vol, safe-haven bond rally, elevated dispersion)
      2: Liquidity Crisis (Sharp across-the-board selloffs, extreme vol, correlations spike to 1.0)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_universe(
        self,
        symbols: List[str],
        n_bars: int = 1000,
        start_date: datetime = datetime(2020, 1, 1),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates aligned OHLCV data for multiple assets and returns:
        (bars_df, true_regimes_df)
        """
        n_assets = len(symbols)
        transition_matrix = np.array([
            [0.97, 0.025, 0.005],  # Low-vol bull is sticky
            [0.05, 0.920, 0.030],  # High-vol bear
            [0.10, 0.200, 0.700],  # Crisis is transient but sharp
        ])

        # Regime parameters per asset: mean daily returns and daily standard deviations
        regime_params = {
            0: {"drift": 0.0006, "vol": 0.008, "corr": 0.25},   # ~15% ann return, ~12% ann vol
            1: {"drift": -0.0004, "vol": 0.018, "corr": 0.55},  # -10% ann return, ~28% ann vol
            2: {"drift": -0.0030, "vol": 0.035, "corr": 0.85},  # -50% ann return, ~55% ann vol
        }

        # Simulate Markov state sequence
        states = np.zeros(n_bars, dtype=int)
        current_state = 0
        for t in range(n_bars):
            states[t] = current_state
            current_state = self.rng.choice(3, p=transition_matrix[current_state])

        dates = [start_date + timedelta(days=i) for i in range(n_bars)]
        # Filter to business days
        trading_dates = [d for d in dates if d.weekday() < 5]
        actual_bars = len(trading_dates)
        states = states[:actual_bars]

        all_bars = []
        for asset_idx, symbol in enumerate(symbols):
            # Asset specific beta / sensitivity
            beta = 0.8 + 0.4 * (asset_idx / max(1, n_assets - 1))
            price = 100.0
            
            for t in range(actual_bars):
                st = states[t]
                drift = regime_params[st]["drift"] * beta
                vol = regime_params[st]["vol"] * beta
                
                # Student-t innovation for fat tails
                ret = self.rng.standard_t(df=5) * vol + drift
                ret = max(min(ret, 0.15), -0.15)  # Clip extremes
                
                open_p = price
                close_p = price * (1.0 + ret)
                high_p = max(open_p, close_p) * (1.0 + abs(self.rng.normal(0, vol * 0.4)))
                low_p = min(open_p, close_p) * (1.0 - abs(self.rng.normal(0, vol * 0.4)))
                volume = float(self.rng.lognormal(mean=14.0, sigma=0.5))

                all_bars.append({
                    "timestamp": trading_dates[t],
                    "symbol": symbol,
                    "open": round(open_p, 4),
                    "high": round(high_p, 4),
                    "low": round(low_p, 4),
                    "close": round(close_p, 4),
                    "volume": round(volume, 0),
                })
                price = close_p

        bars_df = pd.DataFrame(all_bars)
        regimes_df = pd.DataFrame({
            "timestamp": trading_dates,
            "true_regime": states
        })
        return bars_df, regimes_df

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        n_days = (end_date - start_date).days
        bars_df, _ = self.generate_universe([symbol], n_bars=max(10, n_days), start_date=start_date)
        return bars_df
