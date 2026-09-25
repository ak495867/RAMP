"""
Yahoo Finance historical market data collector.
"""

from datetime import datetime
import pandas as pd
import yfinance as yf
from ramp.data.collectors.base import BaseCollector
from ramp.data.validator import MarketDataValidator

class YahooDataCollector(BaseCollector):
    """Fetches equities, ETFs, FX, and crypto data using yfinance."""

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetches historical bars and normalizes column schema:
        ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        """

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=timeframe,
            auto_adjust=True
        )

        if df.empty:
            raise ValueError(f"No data returned from Yahoo for {symbol} between {start_date} and {end_date}")

        df = df.reset_index()

        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
        elif pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df["timestamp"] = df["timestamp"].dt.normalize()

        df["symbol"] = symbol
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

        import numpy as np
        df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
        df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

        df = df.sort_values("timestamp").reset_index(drop=True)
        MarketDataValidator.assert_valid(df, symbol)
        return df
