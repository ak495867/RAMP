from typing import Dict, List, Tuple
import numpy as np


class AlmgrenChrissExecutionOptimizer:

    def __init__(
        self,
        risk_aversion: float = 1e-5,
        temp_impact_eta: float = 2.5e-6,
        perm_impact_gamma: float = 1.0e-6
    ):
        self.risk_aversion = risk_aversion
        self.eta = temp_impact_eta
        self.gamma = perm_impact_gamma

    def compute_optimal_trajectory(
        self,
        total_shares: float,
        time_horizon_hours: float = 6.5,
        n_steps: int = 13,
        daily_volatility: float = 0.015,
        mid_price: float = 100.0
    ) -> List[Dict[str, float]]:
        if total_shares <= 0 or n_steps <= 1:
            return [{"step": 0, "shares_to_trade": total_shares, "remaining_shares": 0.0}]

        tau = time_horizon_hours / n_steps
        sigma_dollar = daily_volatility * mid_price / np.sqrt(6.5)

        half_term = (self.risk_aversion * (sigma_dollar ** 2) * (tau ** 2)) / (2.0 * self.eta)
        arg = half_term + 1.0
        kappa = np.arccosh(max(arg, 1.0 + 1e-9)) / max(tau, 1e-4)

        T = time_horizon_hours
        t_grid = np.linspace(0, T, n_steps + 1)

        holdings = []
        denom = np.sinh(kappa * T)
        for t_j in t_grid:
            if denom > 1e-8:
                x_j = (np.sinh(kappa * (T - t_j)) / denom) * total_shares
            else:
                x_j = total_shares * (1.0 - t_j / T)
            holdings.append(max(float(x_j), 0.0))

        schedule = []
        for j in range(1, len(holdings)):
            trade_qty = holdings[j - 1] - holdings[j]
            schedule.append({
                "step": j,
                "shares_to_trade": round(float(trade_qty), 4),
                "remaining_shares": round(float(holdings[j]), 4),
                "pct_of_order": round(float(trade_qty / total_shares * 100.0), 2)
            })

        return schedule
