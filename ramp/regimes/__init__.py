"""
RAMP Market Regime Detection Module.
"""

from ramp.regimes.base import BaseRegimeDetector
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.bocpd import BayesianOnlineChangePointDetector
from ramp.regimes.filter import RegimeHysteresisFilter

__all__ = [
    "BaseRegimeDetector",
    "OnlineHamiltonFilterHMM",
    "BayesianOnlineChangePointDetector",
    "RegimeHysteresisFilter",
]
