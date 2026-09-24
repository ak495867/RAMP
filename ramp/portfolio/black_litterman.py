"""
Regime-Conditioned Black-Litterman Model.
Dynamically adjusts prior risk-aversion, covariance matrices, and view uncertainties
based on causal regime classifications.
"""

from typing import Dict, List, Tuple
import numpy as np
from ramp.core.types import RegimeState, SignalView


class RegimeConditionedBlackLitterman:
    """
    Implements Black-Litterman allocation conditioned on market regime state.
    
    Regime effects:
      - Low-Vol Bull: High signal confidence, lower prior risk aversion (delta=2.5)
      - High-Vol Bear: Balanced signal weighting, elevated risk aversion (delta=4.5)
      - Crisis/Shock: Views are heavily discounted, high risk aversion (delta=7.0),
                      flight to cash and defensive assets.
    """

    def __init__(self, tau: float = 0.05):
        self.tau = tau
        self.regime_delta_map = {
            0: 2.5,  # Low-vol expansion: normal risk aversion
            1: 4.5,  # High-vol contraction: conservative
            2: 7.5,  # Crisis: extreme risk aversion
        }

    def compute_posterior(
        self,
        symbols: List[str],
        covariance: np.ndarray,
        views: Dict[str, SignalView],
        regime: RegimeState,
        benchmark_weights: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes posterior expected returns mu_post and posterior covariance V_post.
        """
        N = len(symbols)
        if benchmark_weights is None:
            w_mkt = np.ones(N) / N
        else:
            w_mkt = benchmark_weights.copy()

        # Regime-conditioned risk aversion delta
        delta = self.regime_delta_map.get(regime.regime_id, 3.5)

        # 1. Equilibrium Prior Returns: \Pi = \delta * \Sigma * w_mkt
        pi = delta * (covariance @ w_mkt)

        # 2. Build View Matrices P, Q, and \Omega
        # Filter symbols that have active views
        active_indices = []
        q_list = []
        omega_diag = []

        # In crisis regime, discount active views by 70% to prevent chasing false reversals
        regime_discount = 0.30 if regime.regime_id == 2 else 1.0

        for i, sym in enumerate(symbols):
            if sym in views:
                v = views[sym]
                active_indices.append(i)
                q_list.append(v.expected_return * regime_discount)
                # View variance is inversely proportional to confidence: var = tau * sigma^2 / confidence
                asset_var = covariance[i, i]
                conf = max(v.confidence * regime_discount, 0.05)
                omega_ii = (self.tau * asset_var) / conf
                omega_diag.append(omega_ii)

        if not active_indices:
            # No views available, posterior is simply prior
            return pi, covariance

        K = len(active_indices)
        P = np.zeros((K, N))
        for k_idx, i in enumerate(active_indices):
            P[k_idx, i] = 1.0

        Q = np.array(q_list)
        Omega = np.diag(omega_diag)

        # 3. Black-Litterman Closed-Form Posterior Formulation
        # M_inv = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]
        tau_Sigma = self.tau * covariance
        tau_Sigma_inv = np.linalg.pinv(tau_Sigma)
        Omega_inv = np.linalg.pinv(Omega)

        M_inv = tau_Sigma_inv + P.T @ Omega_inv @ P
        M = np.linalg.pinv(M_inv)

        mu_post = M @ (tau_Sigma_inv @ pi + P.T @ Omega_inv @ Q)
        V_post = covariance + M

        return mu_post, V_post
