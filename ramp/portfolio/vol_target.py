"""
Realized Volatility Targeting Engine.
Dynamically scales portfolio leverage and cash buffer to maintain a constant risk profile.
"""

from typing import Tuple
import numpy as np


class VolatilityTargetingEngine:
    """
    Enforces a constant target annualized volatility (e.g. 10%).
    During high-volatility regimes or crisis spikes, scales down risky exposure into cash.
    """

    def __init__(self, target_annual_vol: float = 0.10, max_leverage: float = 1.20):
        self.target_annual_vol = target_annual_vol
        self.max_leverage = max_leverage

    def apply_vol_target(
        self,
        weights: np.ndarray,
        covariance: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Calculates ex-ante volatility and scales weights.
        Returns: (scaled_weights, cash_weight, portfolio_vol)
        """
        N = len(weights)
        if np.all(weights == 0.0):
            return weights, 1.0, 0.0

        # Portfolio ex-ante variance: w^T * Sigma * w * 252
        portfolio_var = float(weights.T @ covariance @ weights) * 252.0
        portfolio_vol = np.sqrt(max(portfolio_var, 1e-8))

        if portfolio_vol <= 1e-4:
            return weights, float(1.0 - np.sum(weights)), 0.0

        # Target scaling factor
        scale = self.target_annual_vol / portfolio_vol
        scale = min(scale, self.max_leverage)

        scaled_weights = weights * scale

        # Ensure gross weights do not exceed leverage limit
        gross_w = np.sum(np.abs(scaled_weights))
        if gross_w > self.max_leverage:
            scaled_weights *= (self.max_leverage / gross_w)

        cash_weight = float(max(0.0, 1.0 - np.sum(scaled_weights)))
        return scaled_weights, cash_weight, float(portfolio_vol * scale)
