"""
Bayesian Online Change-Point Detection (BOCPD - Adams & MacKay 2007).
Recursively computes posterior distribution over run-lengths r_t.
Detects instantaneous structural regime shifts without look-ahead bias.
"""

from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import scipy.stats as stats
from ramp.core.types import RegimeState
from ramp.regimes.base import BaseRegimeDetector


class BayesianOnlineChangePointDetector(BaseRegimeDetector):
    """
    Online Bayesian Change-Point Detection with constant hazard rate.
    P(r_t | x_{1:t}) gives posterior probability that current regime has lasted r_t steps.
    A sudden concentration of probability at r_t = 0 flags a structural break.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 100.0,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        max_run_length: int = 500
    ):
        super().__init__(n_regimes=2, regime_names={0: "stable_regime", 1: "structural_break"})
        self.hazard_rate = hazard_rate
        self.max_run_length = max_run_length

        # Prior hyperparameters for Normal-Inverse-Gamma conjugate model
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Sufficient statistics arrays
        self.mu_t = np.array([mu_0])
        self.kappa_t = np.array([kappa_0])
        self.alpha_t = np.array([alpha_0])
        self.beta_t = np.array([beta_0])

        # Run-length posterior distribution: initially P(r_0 = 0) = 1
        self.R = np.array([1.0])

    def fit(self, features) -> "BayesianOnlineChangePointDetector":
        """Calibrates prior hyperparameters using empirical mean and variance."""
        vals = np.asarray(features).reshape(-1)
        if len(vals) > 0:
            self.mu_0 = float(np.mean(vals))
            var = float(np.var(vals))
            self.beta_0 = max(var, 1e-4)
            self.mu_t = np.array([self.mu_0])
            self.kappa_t = np.array([self.kappa_0])
            self.alpha_t = np.array([self.alpha_0])
            self.beta_t = np.array([self.beta_0])
            self.R = np.array([1.0])
        self.is_fitted = True
        return self

    def filter_step(self, current_features: np.ndarray, timestamp: datetime) -> RegimeState:
        """
        Processes new scalar or first-component observation x_t.
        Returns RegimeState where regime 1 indicates structural break detected (P(r_t=0) is elevated).
        """
        x = float(np.asarray(current_features).reshape(-1)[0])

        # 1. Predictive distribution: Student-t distribution for NIG conjugate prior
        df = 2.0 * self.alpha_t
        loc = self.mu_t
        scale = np.sqrt(self.beta_t * (self.kappa_t + 1.0) / (self.alpha_t * self.kappa_t))
        pred_probs = stats.t.pdf(x, df=df, loc=loc, scale=scale)
        pred_probs = np.maximum(pred_probs, 1e-12)

        # 2. Calculate growth probabilities and changepoint probability
        H = self.hazard_rate
        growth_probs = self.R * pred_probs * (1.0 - H)
        cp_prob = np.sum(self.R * pred_probs * H)

        # 3. Form new run-length posterior
        new_R = np.empty(len(self.R) + 1)
        new_R[0] = cp_prob
        new_R[1:] = growth_probs

        # Normalize
        total = np.sum(new_R)
        if total > 0:
            new_R /= total
        else:
            new_R = np.ones_like(new_R) / len(new_R)

        # Truncate if exceeding max_run_length
        if len(new_R) > self.max_run_length:
            new_R = new_R[:self.max_run_length]
            new_R /= np.sum(new_R)

        self.R = new_R

        # 4. Update sufficient statistics
        new_mu = np.empty(len(self.R))
        new_kappa = np.empty(len(self.R))
        new_alpha = np.empty(len(self.R))
        new_beta = np.empty(len(self.R))

        # Reset state at r=0
        new_mu[0] = self.mu_0
        new_kappa[0] = self.kappa_0
        new_alpha[0] = self.alpha_0
        new_beta[0] = self.beta_0

        # Growth states r > 0
        old_k = self.kappa_t[:len(self.R) - 1]
        old_m = self.mu_t[:len(self.R) - 1]
        old_a = self.alpha_t[:len(self.R) - 1]
        old_b = self.beta_t[:len(self.R) - 1]

        new_kappa[1:] = old_k + 1.0
        new_mu[1:] = (old_k * old_m + x) / (old_k + 1.0)
        new_alpha[1:] = old_a + 0.5
        new_beta[1:] = old_b + (old_k * (x - old_m) ** 2) / (2.0 * (old_k + 1.0))

        self.mu_t = new_mu
        self.kappa_t = new_kappa
        self.alpha_t = new_alpha
        self.beta_t = new_beta

        # Changepoint probability is probability mass at small run length (r <= 2)
        break_prob = float(np.sum(self.R[:min(3, len(self.R))]))
        break_prob = min(max(break_prob, 0.0), 1.0)
        stable_prob = 1.0 - break_prob

        regime_id = 1 if break_prob > 0.4 else 0
        prob_dict = {0: stable_prob, 1: break_prob}

        return RegimeState(
            timestamp=timestamp,
            regime_id=regime_id,
            regime_name=self.regime_names[regime_id],
            probabilities=prob_dict,
            is_transition=(regime_id == 1),
            entropy=float(-break_prob * np.log(max(break_prob, 1e-12)) - stable_prob * np.log(max(stable_prob, 1e-12)))
        )
