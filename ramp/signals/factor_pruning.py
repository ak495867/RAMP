from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats


class DynamicFactorPruner:

    def __init__(self, min_ir_threshold: float = 0.20, lookback_bars: int = 63):
        self.min_ir = min_ir_threshold
        self.lookback = lookback_bars

    @staticmethod
    def calculate_ic(predictions: np.ndarray, forward_returns: np.ndarray) -> float:
        if len(predictions) < 3:
            return 0.0
        corr, _ = stats.spearmanr(predictions, forward_returns)
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def evaluate_factor_health(
        self,
        historical_signals: pd.DataFrame,
        historical_forward_returns: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        health_report = {}
        factors = historical_signals.columns

        for factor in factors:
            if factor not in historical_forward_returns.columns:
                continue

            sig = historical_signals[factor].values[-self.lookback:]
            fwd = historical_forward_returns[factor].values[-self.lookback:]

            valid = ~(np.isnan(sig) | np.isnan(fwd))
            sig_clean = sig[valid]
            fwd_clean = fwd[valid]

            if len(sig_clean) < 10:
                health_report[factor] = {"mean_ic": 0.0, "ic_std": 1.0, "ir": 0.0, "is_active": False}
                continue

            rolling_ics = []
            chunk_size = 10
            for k in range(0, len(sig_clean) - chunk_size, 5):
                ic_k = self.calculate_ic(sig_clean[k:k+chunk_size], fwd_clean[k:k+chunk_size])
                rolling_ics.append(ic_k)

            if rolling_ics:
                mean_ic = float(np.mean(rolling_ics))
                std_ic = float(np.std(rolling_ics, ddof=1)) + 1e-6
                ir = mean_ic / std_ic
            else:
                mean_ic = self.calculate_ic(sig_clean, fwd_clean)
                std_ic = 1.0
                ir = mean_ic

            is_active = (ir >= self.min_ir and mean_ic > 0.0)
            health_report[factor] = {
                "mean_ic": round(mean_ic, 4),
                "ic_std": round(std_ic, 4),
                "ir": round(ir, 4),
                "is_active": is_active
            }

        return health_report

    def compute_pruned_factor_weights(
        self,
        base_weights: Dict[str, float],
        factor_health: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        pruned_weights = {}
        for factor, base_w in base_weights.items():
            health = factor_health.get(factor, {"is_active": True, "ir": 0.5})
            if health.get("is_active", True):
                ir = max(health.get("ir", 0.5), 0.05)
                pruned_weights[factor] = base_w * ir
            else:
                pruned_weights[factor] = 0.0

        total = sum(pruned_weights.values())
        if total > 0:
            return {k: round(v / total, 4) for k, v in pruned_weights.items()}
        return base_weights
