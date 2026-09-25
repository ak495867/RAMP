"""
Abstract base class for quantitative alpha signals.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from ramp.core.types import SignalView

class BaseSignal(ABC):
    """
    Abstract interface for cross-asset alpha signals.
    Outputs standardized SignalView per symbol with expected return and view confidence.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_views(
        self,
        bars_history: pd.DataFrame,
        as_of_date: datetime,
        symbols: List[str]
    ) -> Dict[str, SignalView]:
        """
        Generates views for all symbols as of date T (observing only data <= T).
        Returns Dict[symbol -> SignalView].
        """
        pass
