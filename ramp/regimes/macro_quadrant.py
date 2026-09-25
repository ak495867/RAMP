from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from ramp.core.types import RegimeState
from ramp.regimes.base import BaseRegimeDetector

class MacroQuadrantClassifier(BaseRegimeDetector):

    def __init__(self, growth_lookback: int = 63, inflation_lookback: int = 63):
        super().__init__(
            n_regimes=4,
            regime_names={
                0: "rising_growth_falling_inflation",
                1: "rising_growth_rising_inflation",
                2: "falling_growth_rising_inflation",
                3: "falling_growth_falling_inflation"
            }
        )
        self.growth_lookback = growth_lookback
        self.inflation_lookback = inflation_lookback
        self.growth_threshold: float = 0.0
        self.inflation_threshold: float = 0.0

    def fit(self, features: pd.DataFrame) -> "MacroQuadrantClassifier":
        if "growth_indicator" in features.columns and "inflation_indicator" in features.columns:
            self.growth_threshold = float(features["growth_indicator"].median())
            self.inflation_threshold = float(features["inflation_indicator"].median())
        self.is_fitted = True
        return self

    def filter_step(self, current_features: np.ndarray, timestamp: datetime) -> RegimeState:
        feat = np.asarray(current_features).reshape(-1)
        growth_val = float(feat[0])
        inflation_val = float(feat[1]) if len(feat) > 1 else 0.0

        is_rising_growth = growth_val >= self.growth_threshold
        is_rising_inflation = inflation_val >= self.inflation_threshold

        if is_rising_growth and not is_rising_inflation:
            regime_id = 0
        elif is_rising_growth and is_rising_inflation:
            regime_id = 1
        elif not is_rising_growth and is_rising_inflation:
            regime_id = 2
        else:
            regime_id = 3

        probs = {i: 0.05 for i in range(4)}
        probs[regime_id] = 0.85

        return RegimeState(
            timestamp=timestamp,
            regime_id=regime_id,
            regime_name=self.regime_names[regime_id],
            probabilities=probs,
            is_transition=False,
            entropy=float(-0.85 * np.log(0.85) - 3 * 0.05 * np.log(0.05))
        )

class YieldCurvePCA:

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False

    def fit(self, yield_curve_df: pd.DataFrame) -> "YieldCurvePCA":
        self.pca.fit(yield_curve_df.dropna().values)
        self.is_fitted = True
        return self

    def transform(self, current_yields: np.ndarray) -> Dict[str, float]:
        y = np.asarray(current_yields).reshape(1, -1)
        components = self.pca.transform(y)[0]
        return {
            "level": float(components[0]),
            "slope": float(components[1]) if len(components) > 1 else 0.0,
            "curvature": float(components[2]) if len(components) > 2 else 0.0,
            "explained_variance_ratio": [float(x) for x in self.pca.explained_variance_ratio_]
        }
