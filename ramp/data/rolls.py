"""
Continuous futures roll calculation and backward adjustment (Panama Canal and Ratio methods).
Computes futures roll yield for cross-asset carry factors.
"""

from typing import List, Dict, Literal
import pandas as pd
import numpy as np


class ContinuousFuturesBuilder:
    """
    Stitches individual futures contract slices into a continuous historical series.
    Provides backward-adjustment to prevent artificial gap anomalies on roll dates.
    """

    @staticmethod
    def calculate_roll_yield(front_price: float, next_price: float, days_to_expiry: int) -> float:
        """
        Calculates annualized roll yield / basis:
        (F_front - F_next) / F_front * (365 / days_to_expiry)
        Positive = Backwardation (roll yield is positive for longs)
        Negative = Contango (negative roll yield / headwind for longs)
        """
        if front_price <= 0 or next_price <= 0 or days_to_expiry <= 0:
            return 0.0
        basis = (front_price - next_price) / front_price
        return float(basis * (365.0 / days_to_expiry))

    @staticmethod
    def adjust_continuous_series(
        contracts_df: pd.DataFrame,
        roll_dates: List[pd.Timestamp],
        method: Literal["ratio", "panama_canal"] = "ratio"
    ) -> pd.DataFrame:
        """
        Adjusts continuous price series backward from current active contract.
        
        - 'ratio': Multiplies historical prices by ratio (P_next / P_front) at roll date.
                   Preserves percentage returns. Preferred for equity/financial futures.
        - 'panama_canal': Adds cumulative difference (P_next - P_front) backwards.
                          Preserves absolute dollar spreads. Used for commodities/rates.
        """
        df = contracts_df.copy().sort_values("timestamp")
        if "close" not in df.columns:
            raise ValueError("Dataframe must contain 'close' column")

        df["adj_close"] = df["close"].copy()
        
        # Sort roll dates descending (from most recent back into history)
        sorted_rolls = sorted(roll_dates, reverse=True)

        cumulative_ratio = 1.0
        cumulative_diff = 0.0

        for roll_dt in sorted_rolls:
            # Locate rows at or before the roll date
            mask = df["timestamp"] <= roll_dt
            # Calculate gap at roll date if both front and next contract prices exist
            # Here roll_gap is expected in df or computed between consecutive contracts
            if "roll_gap" in df.columns and roll_dt in df["timestamp"].values:
                gap = df.loc[df["timestamp"] == roll_dt, "roll_gap"].values[0]
                front_p = df.loc[df["timestamp"] == roll_dt, "close"].values[0]
                
                if method == "ratio" and front_p > 0:
                    multiplier = (front_p + gap) / front_p
                    cumulative_ratio *= multiplier
                    df.loc[mask, "adj_close"] *= multiplier
                elif method == "panama_canal":
                    cumulative_diff += gap
                    df.loc[mask, "adj_close"] += gap

        return df
