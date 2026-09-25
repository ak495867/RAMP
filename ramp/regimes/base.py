"""
Abstract base class for causal market regime detectors.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ramp.core.types import RegimeState

class BaseRegimeDetector(ABC):
    """
    Abstract interface for online causal regime detectors.
    Crucial: Models must support online forward-filtering (step-by-step)
    without backward smoothing passes to eliminate look-ahead bias.
    """

    def __init__(self, n_regimes: int = 3, regime_names: Optional[Dict[int, str]] = None):
        self.n_regimes = n_regimes
        self.regime_names = regime_names or {
            0: "low_vol_expansion",
            1: "high_vol_contraction",
            2: "crisis_liquidity_crunch"
        }
        self.is_fitted = False

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> "BaseRegimeDetector":
        """
        Calibrates initial transition and emission priors on historical burn-in window.
        """
        pass

    @abstractmethod
    def filter_step(self, current_features: np.ndarray, timestamp: datetime) -> RegimeState:
        """
        Updates the causal state belief P(S_t | x_{1:t}) for a single new observation.
        """
        pass
