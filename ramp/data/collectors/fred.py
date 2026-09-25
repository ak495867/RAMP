"""
FRED (Federal Reserve Economic Data) macro indicator collector.
Fetches yield curve slopes, inflation expectations, and liquidity indicators.
"""

from datetime import datetime
import pandas as pd
import requests
from ramp.data.collectors.base import BaseCollector

class FREDCollector(BaseCollector):
    """
    Fetches macroeconomic series from St. Louis FRED via direct public CSV endpoints.
    No API key required for public series.
    """

    FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    SERIES_MAP = {
        "YIELD_CURVE_10Y_2Y": "T10Y2Y",                                      
        "VIX": "VIXCLS",                                         
        "FED_FUNDS": "DFF",                                             
        "BREAKEVEN_10Y": "T10YIE",                                          
        "BAA10Y_SPREAD": "BAA10Y",                                                                   
    }

    def fetch_macro_series(self, series_name: str) -> pd.DataFrame:
        """Fetches macro series by name or FRED series ID."""
        series_id = self.SERIES_MAP.get(series_name, series_name)
        url = self.FRED_CSV_URL.format(series_id=series_id)

        response = requests.get(url, timeout=15)
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch FRED series {series_id}, HTTP {response.status_code}")

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = ["timestamp", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().sort_values("timestamp").reset_index(drop=True)
        df["series"] = series_name
        return df

    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        df = self.fetch_macro_series(symbol)
        df = df[(df["timestamp"] >= pd.to_datetime(start_date)) & (df["timestamp"] <= pd.to_datetime(end_date))]

        df["symbol"] = symbol
        df["open"] = df["value"]
        df["high"] = df["value"]
        df["low"] = df["value"]
        df["close"] = df["value"]
        df["volume"] = 0.0
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
