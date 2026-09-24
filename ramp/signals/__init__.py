"""
RAMP Alpha Signal Library.
"""

from ramp.signals.base import BaseSignal
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.signals.mean_reversion import MeanReversionSignal
from ramp.signals.vrp import VolatilityRiskPremiumSignal

__all__ = [
    "BaseSignal",
    "TimeSeriesMomentumSignal",
    "CrossAssetCarrySignal",
    "MeanReversionSignal",
    "VolatilityRiskPremiumSignal",
]
