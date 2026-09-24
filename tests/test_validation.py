"""
Unit tests for Combinatorial Purged Cross-Validation, DSR, and PBO.
"""

import numpy as np
import pytest
from ramp.validation.deflated_sharpe import DeflatedSharpeRatio
from ramp.validation.cpcv import CombinatorialPurgedCV
from ramp.validation.pbo import ProbabilityOfBacktestOverfitting


def test_deflated_sharpe_ratio():
    # If observed SR is high and only 1 trial, DSR should be high
    dsr_single = DeflatedSharpeRatio.deflated_sharpe_ratio(
        observed_sr=2.0,
        trials_variance=0.25,
        num_trials=1,
        n_observations=500
    )
    assert dsr_single > 0.95

    # If 1000 trials were run and trials variance is high, DSR should be heavily deflated
    dsr_overfit = DeflatedSharpeRatio.deflated_sharpe_ratio(
        observed_sr=1.2,
        trials_variance=0.5,
        num_trials=1000,
        n_observations=252
    )
    assert dsr_overfit < dsr_single


def test_combinatorial_purged_cv_split():
    cpcv = CombinatorialPurgedCV(n_groups=5, k_test_groups=2, embargo_pct=0.02)
    n_samples = 200

    splits = list(cpcv.split(n_samples))
    # 5 choose 2 = 10 combinations
    assert len(splits) == 10

    for train_idx, test_idx in splits:
        # Train and test should never intersect
        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(test_idx) > 0
        assert len(train_idx) > 0


def test_probability_of_backtest_overfitting():
    rng = np.random.default_rng(42)
    # Generate 100 random noise strategies (none have true alpha)
    noise_returns = rng.normal(0.0001, 0.01, size=(500, 20))
    
    pbo_result = ProbabilityOfBacktestOverfitting.compute_pbo(noise_returns, n_slices=6)
    assert "pbo" in pbo_result
    # For random noise strategies, PBO should be close to 0.50 (coin toss)
    assert 0.20 <= pbo_result["pbo"] <= 0.80
