"""
Comprehensive Institutional Quantitative Performance & Risk Metrics.
Calculates CAGR, Sharpe, Sortino, Calmar, MaxDD, VaR, CVaR, Turnover, and Omega Ratio.
"""

from typing import Dict
import numpy as np
import pandas as pd


class PerformanceMetricsCalculator:
    """
    Computes rigorous institutional risk and performance metrics on daily returns.
    """

    @staticmethod
    def calculate_all(
        daily_returns: pd.Series,
        risk_free_rate_annual: float = 0.045,
        turnover_annual: float = 0.0
    ) -> Dict[str, float]:
        """Calculates comprehensive metrics suite."""
        rets = daily_returns.dropna().values
        if len(rets) < 5:
            return {"error": "Insufficient return observations"}

        daily_rf = risk_free_rate_annual / 252.0
        excess_rets = rets - daily_rf

        # Cumulative NAV & Drawdown
        cum_ret = np.cumprod(1.0 + rets)
        running_max = np.maximum.accumulate(cum_ret)
        drawdowns = (cum_ret - running_max) / running_max
        max_dd = float(np.min(drawdowns))

        # CAGR
        n_days = len(rets)
        years = max(n_days / 252.0, 0.01)
        total_ret = cum_ret[-1] - 1.0
        cagr = float((1.0 + total_ret) ** (1.0 / years) - 1.0)

        # Annualized Volatility
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0))

        # Sharpe Ratio
        mean_excess = float(np.mean(excess_rets))
        sharpe = float((mean_excess / max(np.std(rets, ddof=1), 1e-6)) * np.sqrt(252.0))

        # Downside Deviation & Sortino Ratio
        downside_rets = rets[rets < 0.0]
        if len(downside_rets) > 0:
            downside_std = float(np.std(downside_rets, ddof=1) * np.sqrt(252.0))
            sortino = float(cagr / max(downside_std, 1e-6))
        else:
            downside_std = 0.0
            sortino = float("inf")

        # Calmar Ratio
        calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0

        # Hit Rate & Profit Factor
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        hit_rate = float(len(wins) / len(rets))
        gross_profits = np.sum(wins) if len(wins) > 0 else 0.0
        gross_losses = abs(np.sum(losses)) if len(losses) > 0 else 1e-6
        profit_factor = float(gross_profits / gross_losses)

        # Value at Risk (VaR 95%) & Conditional VaR / Expected Shortfall (CVaR 95%)
        var_95 = float(-np.percentile(rets, 5))
        cvar_95 = float(-np.mean(rets[rets <= -var_95])) if any(rets <= -var_95) else var_95

        # Omega Ratio (threshold = daily risk-free rate)
        gains = excess_rets[excess_rets > 0]
        disadvantages = abs(excess_rets[excess_rets < 0])
        omega = float(np.sum(gains) / max(np.sum(disadvantages), 1e-6))

        return {
            "cagr": round(cagr, 4),
            "annualized_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown": round(max_dd, 4),
            "calmar_ratio": round(calmar, 3),
            "hit_rate": round(hit_rate, 4),
            "profit_factor": round(profit_factor, 3),
            "var_95_daily": round(var_95, 4),
            "cvar_95_daily": round(cvar_95, 4),
            "omega_ratio": round(omega, 3),
            "annualized_turnover": round(turnover_annual, 3),
            "total_trading_days": n_days,
        }
