"""
Abstract base class for data collectors.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
import pandas as pd


class BaseCollector(ABC):
    """Abstract interface for all asset and macro data collectors."""

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetches historical bars for a symbol.
        Returns a DataFrame with columns:
        ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        """
        pass
