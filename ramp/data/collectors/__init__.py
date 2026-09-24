"""
Data collectors for RAMP.
"""

from ramp.data.collectors.base import BaseCollector
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.data.collectors.yahoo import YahooDataCollector
from ramp.data.collectors.fred import FREDCollector

__all__ = [
    "BaseCollector",
    "SyntheticRegimeDataGenerator",
    "YahooDataCollector",
    "FREDCollector",
]
