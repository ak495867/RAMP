from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class BarraFactorRiskModel:

    def __init__(self, factor_names: Optional[List[str]] = None):
        self.factor_names = factor_names or [
            "equity_market_beta",
            "rates_duration",
            "credit_spread",
            "commodity_inflation",
            "momentum_style",
            "value_style",
            "crypto_liquidity"
        ]

    def estimate_factor_model(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        aligned = asset_returns.join(factor_returns, how="inner").dropna()
        asset_cols = list(asset_returns.columns)
        factor_cols = list(factor_returns.columns)

        Y = aligned[asset_cols].values
        F_matrix = aligned[factor_cols].values

        n_assets = len(asset_cols)
        n_factors = len(factor_cols)

        X_loadings = np.zeros((n_assets, n_factors))
        specific_variances = np.zeros(n_assets)

        F_cov = np.cov(F_matrix, rowvar=False)

        for i in range(n_assets):
            y_i = Y[:, i]
            reg = np.linalg.lstsq(F_matrix, y_i, rcond=None)
            betas = reg[0]
            residuals = y_i - F_matrix @ betas
            X_loadings[i, :] = betas
            specific_variances[i] = float(np.var(residuals, ddof=n_factors))

        Delta_specific = np.diag(specific_variances)
        return X_loadings, F_cov, Delta_specific

    def decompose_portfolio_risk(
        self,
        weights: np.ndarray,
        X_loadings: np.ndarray,
        factor_cov: np.ndarray,
        Delta_specific: np.ndarray
    ) -> Dict[str, float]:
        w = np.asarray(weights).reshape(-1)
        systematic_var = float(w.T @ (X_loadings @ factor_cov @ X_loadings.T) @ w)
        specific_var = float(w.T @ Delta_specific @ w)
        total_var = max(systematic_var + specific_var, 1e-8)

        total_vol = float(np.sqrt(total_var * 252.0))
        systematic_vol = float(np.sqrt(max(systematic_var * 252.0, 0.0)))
        specific_vol = float(np.sqrt(max(specific_var * 252.0, 0.0)))

        systematic_ratio = systematic_var / total_var
        specific_ratio = specific_var / total_var

        factor_exposures = X_loadings.T @ w
        marginal_factor_risk = (factor_cov @ factor_exposures) / np.sqrt(total_var)

        factor_risk_contrib = {}
        for idx, f_name in enumerate(self.factor_names[:len(factor_exposures)]):
            factor_risk_contrib[f_name] = float(factor_exposures[idx] * marginal_factor_risk[idx])

        return {
            "total_annual_vol": round(total_vol, 4),
            "systematic_annual_vol": round(systematic_vol, 4),
            "specific_annual_vol": round(specific_vol, 4),
            "systematic_risk_pct": round(systematic_ratio * 100.0, 2),
            "specific_risk_pct": round(specific_ratio * 100.0, 2),
            "factor_risk_contributions": factor_risk_contrib
        }
