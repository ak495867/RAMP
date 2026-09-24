"""
Data validation and sanity checks for raw and adjusted market data feeds.
"""

from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


class DataValidationError(Exception):
    """Raised when data integrity checks fail."""
    pass


class MarketDataValidator:
    """
    Validates point-in-time correctness, timestamp consistency, and absence of data corruption.
    """

    @staticmethod
    def validate_bars(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validates OHLCV DataFrame for a symbol.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            errors.append(f"Missing required columns for {symbol}: {missing}")
            return False, errors

        # 1. Monotonic timestamps & no duplicates
        timestamps = pd.to_datetime(df["timestamp"])
        if not timestamps.is_monotonic_increasing:
            errors.append(f"Timestamps for {symbol} are not monotonically increasing")
        if timestamps.duplicated().any():
            dup_count = timestamps.duplicated().sum()
            errors.append(f"Found {dup_count} duplicate timestamps for {symbol}")

        # 2. Check for NaN / Nulls
        for col in ["open", "high", "low", "close"]:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                errors.append(f"Found {nan_count} NaNs in {col} for {symbol}")

        # 3. Check for non-positive prices
        for col in ["open", "high", "low", "close"]:
            non_pos = (df[col] <= 0).sum()
            if non_pos > 0:
                errors.append(f"Found {non_pos} non-positive prices in {col} for {symbol}")

        # 4. Check High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            errors.append(f"High < Low in {invalid_hl} rows for {symbol}")

        invalid_h_oc = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
        if invalid_h_oc > 0:
            errors.append(f"High < Open or Close in {invalid_h_oc} rows for {symbol}")

        invalid_l_oc = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
        if invalid_l_oc > 0:
            errors.append(f"Low > Open or Close in {invalid_l_oc} rows for {symbol}")

        # 5. Check for non-negative volume
        neg_vol = (df["volume"] < 0).sum()
        if neg_vol > 0:
            errors.append(f"Found {neg_vol} negative volume values for {symbol}")

        # 6. Check for unadjusted split anomalies (> 5x single-day price jumps without adjustment)
        returns = df["close"].pct_change().dropna()
        extreme_jumps = (returns.abs() > 4.0).sum()
        if extreme_jumps > 0:
            errors.append(f"Found {extreme_jumps} extreme price jumps (>400%) in {symbol} (possible split artifact)")

        return len(errors) == 0, errors

    @classmethod
    def assert_valid(cls, df: pd.DataFrame, symbol: str) -> None:
        """Raises DataValidationError if data fails checks."""
        is_valid, errors = cls.validate_bars(df, symbol)
        if not is_valid:
            raise DataValidationError(f"Validation failed for {symbol}: " + "; ".join(errors))
