"""
Probability of Backtest Overfitting (PBO).
Calculated via Combinatorially Symmetric Cross-Validation (CSCV) - Bailey et al. (2016).
Measures the likelihood that the best in-sample strategy yields below-median out-of-sample performance.
"""

from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np


class ProbabilityOfBacktestOverfitting:
    """
    Evaluates selection bias across a matrix of backtested strategy variants.
    """

    @staticmethod
    def compute_pbo(returns_matrix: np.ndarray, n_slices: int = 8) -> Dict[str, float]:
        """
        Computes PBO from an (T x N) returns matrix:
          T = time periods
          N = number of strategy variants/trials
        """
        T, N = returns_matrix.shape
        if N < 2:
            return {"pbo": 0.0, "message": "At least 2 strategy variants required to calculate PBO"}

        slice_size = T // n_slices
        slices = [returns_matrix[i * slice_size:(i + 1) * slice_size] for i in range(n_slices)]

        k_test = n_slices // 2
        logits = []
        underperform_count = 0
        total_combos = 0

        for test_combo in combinations(range(n_slices), k_test):
            total_combos += 1
            train_combo = [i for i in range(n_slices) if i not in test_combo]

            train_rets = np.concatenate([slices[i] for i in train_combo], axis=0)
            test_rets = np.concatenate([slices[i] for i in test_combo], axis=0)

            # In-sample Sharpe ratio for each model
            is_means = np.mean(train_rets, axis=0)
            is_stds = np.std(train_rets, axis=0, ddof=1)
            is_sharpes = is_means / np.maximum(is_stds, 1e-6)

            best_model_idx = int(np.argmax(is_sharpes))

            # Out-of-sample Sharpe ratios
            oos_means = np.mean(test_rets, axis=0)
            oos_stds = np.std(test_rets, axis=0, ddof=1)
            oos_sharpes = oos_means / np.maximum(oos_stds, 1e-6)

            # Rank of best IS model in OOS
            sorted_oos_ranks = np.argsort(np.argsort(oos_sharpes))
            relative_rank = sorted_oos_ranks[best_model_idx] / (N - 1.0)  # 0.0 (worst) to 1.0 (best)

            if relative_rank < 0.5:
                underperform_count += 1

            # Logit transformation
            rank_clamped = min(max(relative_rank, 1e-4), 1.0 - 1e-4)
            logit = np.log(rank_clamped / (1.0 - rank_clamped))
            logits.append(logit)

        pbo = float(underperform_count / max(total_combos, 1))

        return {
            "pbo": round(pbo, 4),
            "num_combinations": total_combos,
            "mean_oos_rank_percentile": round(float(np.mean([1.0 / (1.0 + np.exp(-l)) for l in logits])), 4),
            "is_overfitted": pbo > 0.40,
        }
