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

    def align_universe_bars(
        self,
        bars_df: pd.DataFrame,
        benchmark_symbol: str = "SPY"
    ) -> pd.DataFrame:
        """
        Aligns a multi-asset universe DataFrame to the benchmark's active trading days.
        For non-benchmark assets (e.g. crypto on weekdays or assets with occasional missing prints),
        forward-fills missing prices to prevent look-ahead bias and drop weekend-only bars.
        """
        df = bars_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        bm_dates = df[df["symbol"] == benchmark_symbol]["timestamp"].drop_duplicates().sort_values()
        if bm_dates.empty:

            bm_dates = df[df["timestamp"].dt.weekday < 5]["timestamp"].drop_duplicates().sort_values()

        symbols = df["symbol"].unique()
        aligned_records = []

        for sym in symbols:
            sym_df = df[df["symbol"] == sym].drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

            sym_df = sym_df.set_index("timestamp").reindex(bm_dates)
            sym_df["symbol"] = sym

            sym_df["close"] = sym_df["close"].ffill().bfill()
            sym_df["open"] = sym_df["open"].fillna(sym_df["close"])
            sym_df["high"] = sym_df["high"].fillna(sym_df["close"])
            sym_df["low"] = sym_df["low"].fillna(sym_df["close"])
            sym_df["volume"] = sym_df["volume"].fillna(0.0)

            sym_df = sym_df.reset_index().rename(columns={"index": "timestamp"})
            aligned_records.append(sym_df)

        aligned_df = pd.concat(aligned_records, ignore_index=True)
        return aligned_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
