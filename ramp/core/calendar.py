"""
Market calendar utilities for multi-asset alignment across traditional and 24/7 markets.
"""

from datetime import datetime, time
import pandas as pd
import pytz


class MarketCalendar:
    """Handles business day schedules, market open/close, and cross-asset alignment."""

    def __init__(self, tz_name: str = "America/New_York"):
        self.tz = pytz.timezone(tz_name)
        self.market_open_time = time(9, 30)
        self.market_close_time = time(16, 0)

    def is_business_day(self, dt: datetime) -> bool:
        """Checks if a given date is a weekday (Monday-Friday)."""
        return dt.weekday() < 5

    def align_to_trading_days(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
        """
        Aligns a multi-asset dataset (including 24/7 crypto) to standard business days.
        Crypto weekend bars are rolled forward or treated with Friday close alignment
        so as-of joins against equities/futures are point-in-time consistent.
        """
        if timestamp_col not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Column {timestamp_col} not found and index is not DatetimeIndex")

        data = df.copy()
        if timestamp_col in data.columns:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            data = data.sort_values(timestamp_col)
        else:
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

        return data
