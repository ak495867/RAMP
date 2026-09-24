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
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=timeframe,
            auto_adjust=False  # Keep raw and dividend adjusted distinct
        )

        if df.empty:
            raise ValueError(f"No data returned from Yahoo for {symbol} between {start_date} and {end_date}")

        df = df.reset_index()
        # Rename columns to lowercase standard
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close"
        })

        # Remove timezone if present to maintain naive UTC timestamps
        if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df["symbol"] = symbol
        df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        
        # Sort and validate
        df = df.sort_values("timestamp").reset_index(drop=True)
        MarketDataValidator.assert_valid(df, symbol)
        return df
