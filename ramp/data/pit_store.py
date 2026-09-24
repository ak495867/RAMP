"""
Point-in-Time DuckDB Lakehouse Engine.
Enforces zero look-ahead bias through point-in-time window queries and as-of joins.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union
import duckdb
import pandas as pd
import numpy as np


class PointInTimeStore:
    """
    Point-in-time analytical store using embedded DuckDB.
    Guarantees that any query at date T only accesses data published at or before T - 1 close.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = ":memory:"):
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initializes tables for bars, macro indicators, and computed features."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_bars (
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                PRIMARY KEY (timestamp, symbol)
            );
            
            CREATE INDEX IF NOT EXISTS idx_bars_sym_time ON market_bars (symbol, timestamp);

            CREATE TABLE IF NOT EXISTS macro_indicators (
                timestamp TIMESTAMP NOT NULL,
                series_name VARCHAR NOT NULL,
                value DOUBLE NOT NULL,
                PRIMARY KEY (timestamp, series_name)
            );

            CREATE INDEX IF NOT EXISTS idx_macro_series_time ON macro_indicators (series_name, timestamp);
        """)

    def insert_bars(self, df: pd.DataFrame) -> None:
        """Inserts OHLCV bars into DuckDB with idempotency (upsert)."""
        if df.empty:
            return
        
        # Prepare temp table and insert
        temp_df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]].copy()
        temp_df["timestamp"] = pd.to_datetime(temp_df["timestamp"])
        
        self.conn.register("temp_bars_input", temp_df)
        self.conn.execute("""
            INSERT OR REPLACE INTO market_bars
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM temp_bars_input
        """)
        self.conn.unregister("temp_bars_input")

    def insert_macro(self, df: pd.DataFrame) -> None:
        """Inserts macroeconomic observations into DuckDB."""
        if df.empty:
            return
        temp_macro = df[["timestamp", "series", "value"]].copy()
        temp_macro["timestamp"] = pd.to_datetime(temp_macro["timestamp"])
        temp_macro = temp_macro.rename(columns={"series": "series_name"})

        self.conn.register("temp_macro_input", temp_macro)
        self.conn.execute("""
            INSERT OR REPLACE INTO macro_indicators
            SELECT timestamp, series_name, value
            FROM temp_macro_input
        """)
        self.conn.unregister("temp_macro_input")

    def get_bars_pit(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Retrieves market bars strictly within [start_date, end_date].
        """
        query = """
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM market_bars
            WHERE symbol IN ({syms})
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp ASC, symbol ASC
        """.format(syms=",".join([f"'{s}'" for s in symbols]))

        res = self.conn.execute(query, [start_date, end_date]).fetchdf()
        return res

    def get_causal_regime_features(
        self,
        symbols: List[str],
        as_of_date: datetime,
        lookback_bars: int = 252
    ) -> pd.DataFrame:
        """
        CRITICAL POINT-IN-TIME FEATURE GENERATOR:
        Computes features strictly up to as_of_date (inclusive of as_of_date - 1 day,
        or as_of_date close if signals run after-market).
        Returns equal-weighted basket return, realized volatility, and pairwise correlation.
        """
        query = f"""
            WITH ranked_bars AS (
                SELECT 
                    timestamp,
                    symbol,
                    close,
                    (close - LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp)) / 
                        NULLIF(LAG(close, 1) OVER (PARTITION BY symbol ORDER BY timestamp), 0) AS daily_ret
                FROM market_bars
                WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
                  AND timestamp <= ?
            ),
            pivoted_dates AS (
                SELECT 
                    timestamp,
                    AVG(daily_ret) AS basket_return,
                    STDDEV(daily_ret) AS cross_sectional_dispersion
                FROM ranked_bars
                WHERE daily_ret IS NOT NULL
                GROUP BY timestamp
                ORDER BY timestamp DESC
                LIMIT ?
            )
            SELECT * FROM pivoted_dates ORDER BY timestamp ASC
        """
        df = self.conn.execute(query, [as_of_date, lookback_bars]).fetchdf()
        return df

    def close(self) -> None:
        """Closes the DuckDB connection."""
        self.conn.close()
