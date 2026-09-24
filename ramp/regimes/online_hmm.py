"""
Causal Online Hamilton Filtering Hidden Markov Model.
Strictly causal: computes forward filtering probabilities P(S_t | x_{1:t}) with zero backward smoothing.
Applies variance-ordered state canonicalization to permanently prevent label switching.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from ramp.core.types import RegimeState
from ramp.regimes.base import BaseRegimeDetector


class OnlineHamiltonFilterHMM(BaseRegimeDetector):
    """
    Online Hidden Markov Model implementing the Hamilton Forward Filter.
    Zero look-ahead bias: step-by-step causal Bayesian updates.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        regime_names: Optional[Dict[int, str]] = None,
        random_state: int = 42
    ):
        super().__init__(n_regimes, regime_names)
        self.random_state = random_state

        # Parameters
        self.means: np.ndarray = np.zeros((n_regimes, 1))
        self.covariances: List[np.ndarray] = [np.eye(1) for _ in range(n_regimes)]
        self.transition_matrix: np.ndarray = np.eye(n_regimes)
        self.filtered_probs: np.ndarray = np.ones(n_regimes) / n_regimes
        self.feature_names: List[str] = []

    def fit(self, features: pd.DataFrame) -> "OnlineHamiltonFilterHMM":
        """
        Fits initial Gaussian emission and transition parameters on burn-in window.
        Enforces volatility ordering (0=low-vol, 1=medium-vol, 2=high-vol) to prevent label switching.
        """
        self.feature_names = list(features.columns)
        X = features.values
        if len(X) < 30:
            raise ValueError(f"Need at least 30 observations to fit HMM priors, got {len(X)}")

        # Fit Gaussian Mixture to initialize emission distributions
        gmm = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            max_iter=150
        )
        labels = gmm.fit_predict(X)

        # Compute empirical variances to canonicalize state ordering
        # Regime 0 = Lowest variance, Regime N-1 = Highest variance (Crisis/Liquidity shock)
        variances = [np.trace(gmm.covariances_[i]) for i in range(self.n_regimes)]
        order = np.argsort(variances)

        self.means = gmm.means_[order]
        self.covariances = [gmm.covariances_[i] for i in order]

        # Re-map labels according to canonical order
        remap = {old: new for new, old in enumerate(order)}
        canonical_labels = np.array([remap[lbl] for lbl in labels])

        # Estimate empirical transition matrix with Laplace smoothing
        A = np.ones((self.n_regimes, self.n_regimes)) * 0.5  # Prior smoothing
        for i in range(len(canonical_labels) - 1):
            A[canonical_labels[i], canonical_labels[i + 1]] += 1.0

        # Normalize row-stochastic matrix
        self.transition_matrix = A / A.sum(axis=1, keepdims=True)

        # Set initial prior state probability to stationary or uniform
        self.filtered_probs = np.ones(self.n_regimes) / self.n_regimes
        self.is_fitted = True
        return self

    def filter_step(self, current_features: np.ndarray, timestamp: datetime) -> RegimeState:
        """
        Executes a single Hamilton Forward Filter step for new observation x_t.
        Returns P(S_t = k | x_{1:t}).
        """
        if not self.is_fitted:
            raise RuntimeError("HMM detector must be fitted before running filter steps.")

        x = np.asarray(current_features, dtype=float).reshape(-1)

        # 1. Prediction step: P(S_t = j | x_{1:t-1}) = \sum_i P(S_{t-1} = i | x_{1:t-1}) * A_{ij}
        prior_state_probs = self.filtered_probs @ self.transition_matrix

        # 2. Emission likelihood: f(x_t | S_t = j)
        likelihoods = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            try:
                # Add small epsilon to diagonal for numerical conditioning
                cov = self.covariances[j] + np.eye(len(x)) * 1e-6
                dist = multivariate_normal(mean=self.means[j], cov=cov, allow_singular=True)
                likelihoods[j] = dist.pdf(x)
            except Exception:
                likelihoods[j] = 1e-8

        # Floor likelihoods to avoid division by zero
        likelihoods = np.maximum(likelihoods, 1e-12)

        # 3. Update step: \alpha_t(j) = likelihood_j * prior_j
        unnormalized = prior_state_probs * likelihoods
        sum_p = np.sum(unnormalized)
        if sum_p <= 0 or np.isnan(sum_p):
            self.filtered_probs = np.ones(self.n_regimes) / self.n_regimes
        else:
            self.filtered_probs = unnormalized / sum_p

        # 4. Calculate Shannon Entropy H = -\sum p log(p) as uncertainty metric
        safe_probs = np.maximum(self.filtered_probs, 1e-12)
        entropy = float(-np.sum(safe_probs * np.log(safe_probs)))

        active_id = int(np.argmax(self.filtered_probs))
        prob_dict = {i: float(self.filtered_probs[i]) for i in range(self.n_regimes)}

        return RegimeState(
            timestamp=timestamp,
            regime_id=active_id,
            regime_name=self.regime_names.get(active_id, f"regime_{active_id}"),
            probabilities=prob_dict,
            is_transition=False,
            entropy=entropy
        )
