"""
Robust Convex Portfolio Optimizer using CVXPY.
Incorporates L1 turnover penalties, gross leverage caps, and asset-class limits.
"""

from typing import Dict, List, Optional
import cvxpy as cp
import numpy as np
from ramp.portfolio.hrp import HierarchicalRiskParity

class RobustConvexOptimizer:
    """
    Convex portfolio optimizer with explicit L1 turnover regularization.
    Penalizes unnecessary portfolio rebalancing to control transaction costs.
    """

    def __init__(
        self,
        risk_aversion_gamma: float = 3.0,
        turnover_penalty_lambda: float = 0.002,
        max_position_weight: float = 0.25,
        max_asset_class_weight: float = 0.50,
        allow_shorting: bool = False,
        gross_leverage_limit: float = 1.0
    ):
        self.gamma = risk_aversion_gamma
        self.turnover_lambda = turnover_penalty_lambda
        self.max_position_weight = max_position_weight
        self.max_asset_class_weight = max_asset_class_weight
        self.allow_shorting = allow_shorting
        self.gross_leverage_limit = gross_leverage_limit

    def optimize(
        self,
        mu: np.ndarray,
        covariance: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        asset_class_map: Optional[Dict[str, List[int]]] = None
    ) -> np.ndarray:
        """
        Solves the convex optimization problem:
        min_w  (gamma/2) * w^T Sigma w - mu^T w + lambda * ||w - w_prev||_1
        """
        N = len(mu)
        if current_weights is None:
            w_prev = np.zeros(N)
        else:
            w_prev = current_weights.copy()

        Sigma = covariance + np.eye(N) * 1e-6

        w = cp.Variable(N)

        risk_term = 0.5 * self.gamma * cp.quad_form(w, Sigma)
        return_term = mu @ w
        turnover_term = self.turnover_lambda * cp.norm1(w - w_prev)

        objective = cp.Minimize(risk_term - return_term + turnover_term)

        constraints = [
            cp.sum(w) <= 1.0,                       
        ]

        if not self.allow_shorting:
            constraints.append(w >= 0.0)
            constraints.append(w <= self.max_position_weight)
        else:
            constraints.append(w >= -self.max_position_weight)
            constraints.append(w <= self.max_position_weight)
            constraints.append(cp.norm1(w) <= self.gross_leverage_limit)

        if asset_class_map:
            for class_name, indices in asset_class_map.items():
                if indices:
                    constraints.append(cp.sum(w[indices]) <= self.max_asset_class_weight)

        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                weights = np.array(w.value).reshape(-1)

                weights = np.where(np.abs(weights) < 1e-4, 0.0, weights)
                return weights
        except Exception:
            pass

        return HierarchicalRiskParity.allocate(covariance)
