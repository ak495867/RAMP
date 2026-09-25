from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import scipy.stats as stats
from ramp.core.types import RegimeState
from ramp.regimes.base import BaseRegimeDetector

class HiddenSemiMarkovModel(BaseRegimeDetector):

    def __init__(
        self,
        n_regimes: int = 3,
        regime_names: Optional[Dict[int, str]] = None,
        expected_durations: Optional[Dict[int, float]] = None
    ):
        super().__init__(n_regimes, regime_names)
        self.durations = expected_durations or {
            0: 200.0,
            1: 60.0,
            2: 15.0
        }
        self.means = np.zeros(n_regimes)
        self.variances = np.ones(n_regimes)
        self.current_state = 0
        self.current_dwell = 1
        self.transition_matrix = np.array([
            [0.0, 0.85, 0.15],
            [0.60, 0.0, 0.40],
            [0.70, 0.30, 0.0]
        ])

    def fit(self, features) -> "HiddenSemiMarkovModel":
        vals = np.asarray(features).reshape(-1)
        if len(vals) >= 30:
            p33 = float(np.percentile(vals, 33))
            p66 = float(np.percentile(vals, 66))
            self.means = np.array([float(np.mean(vals[vals <= p33])), float(np.mean(vals[(vals > p33) & (vals <= p66)])), float(np.mean(vals[vals > p66]))])
            self.variances = np.array([float(np.var(vals[vals <= p33]) + 1e-4), float(np.var(vals[(vals > p33) & (vals <= p66)]) + 1e-4), float(np.var(vals[vals > p66]) + 1e-4)])
        self.is_fitted = True
        return self

    def _state_exit_probability(self, state: int, dwell: int) -> float:
        mu = self.durations.get(state, 50.0)
        p_exit = stats.poisson.cdf(dwell, mu=mu)
        return float(np.clip(p_exit, 0.01, 0.95))

    def filter_step(self, current_features: np.ndarray, timestamp: datetime) -> RegimeState:
        x = float(np.asarray(current_features).reshape(-1)[0])
        likelihoods = np.zeros(self.n_regimes)

        for j in range(self.n_regimes):
            dist = stats.norm(loc=self.means[j], scale=np.sqrt(self.variances[j]))
            likelihoods[j] = max(dist.pdf(x), 1e-8)

        p_exit = self._state_exit_probability(self.current_state, self.current_dwell)

        state_scores = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            if j == self.current_state:
                state_scores[j] = likelihoods[j] * (1.0 - p_exit)
            else:
                p_trans = self.transition_matrix[self.current_state, j]
                state_scores[j] = likelihoods[j] * p_exit * p_trans

        total_score = np.sum(state_scores)
        if total_score > 0:
            probs = state_scores / total_score
        else:
            probs = np.ones(self.n_regimes) / self.n_regimes

        new_state = int(np.argmax(probs))
        if new_state == self.current_state:
            self.current_dwell += 1
            is_trans = False
        else:
            self.current_state = new_state
            self.current_dwell = 1
            is_trans = True

        prob_dict = {i: float(probs[i]) for i in range(self.n_regimes)}
        entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))

        return RegimeState(
            timestamp=timestamp,
            regime_id=self.current_state,
            regime_name=self.regime_names.get(self.current_state, f"regime_{self.current_state}"),
            probabilities=prob_dict,
            is_transition=is_trans,
            entropy=entropy
        )
