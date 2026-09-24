"""
Combinatorial Purged Cross-Validation (CPCV) with Embargoing.
Prevents information leakage from overlapping labels and serial correlation in financial time series.
"""

from itertools import combinations
from typing import Generator, List, Tuple
import numpy as np
import pandas as pd


class CombinatorialPurgedCV:
    """
    CPCV splits N observations into G groups, and selects k groups for testing.
    Purges overlapping training samples and applies an embargo period after test segments.
    """

    def __init__(self, n_groups: int = 6, k_test_groups: int = 2, embargo_pct: float = 0.01):
        if k_test_groups >= n_groups:
            raise ValueError("k_test_groups must be strictly less than n_groups")
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_pct = embargo_pct

    def split(self, n_samples: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for each combination.
        """
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_groups
        groups = []

        for g in range(self.n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < self.n_groups - 1 else n_samples
            groups.append(indices[start:end])

        embargo_bars = max(int(n_samples * self.embargo_pct), 1)

        for test_group_indices in combinations(range(self.n_groups), self.k_test_groups):
            test_indices = np.concatenate([groups[i] for i in test_group_indices])
            
            # Identify test segments to purge and embargo
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # Embargo right after each test group
            for g_idx in test_group_indices:
                test_end = groups[g_idx][-1]
                embargo_end = min(test_end + embargo_bars, n_samples)
                train_mask[test_end:embargo_end] = False

            train_indices = indices[train_mask]
            yield train_indices, test_indices
