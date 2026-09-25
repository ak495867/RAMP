"""
Unit tests for Combinatorial Purged Cross-Validation, DSR, and PBO.
"""

import numpy as np
import pytest
from ramp.validation.deflated_sharpe import DeflatedSharpeRatio
from ramp.validation.cpcv import CombinatorialPurgedCV
from ramp.validation.pbo import ProbabilityOfBacktestOverfitting

def test_deflated_sharpe_ratio():

    dsr_single = DeflatedSharpeRatio.deflated_sharpe_ratio(
        observed_sr=2.0,
        trials_variance=0.25,
        num_trials=1,
        n_observations=500
    )
    assert dsr_single > 0.95

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

    assert len(splits) == 10

    for train_idx, test_idx in splits:

        assert len(set(train_idx).intersection(set(test_idx))) == 0
        assert len(test_idx) > 0
        assert len(train_idx) > 0

def test_probability_of_backtest_overfitting():
    rng = np.random.default_rng(42)

    noise_returns = rng.normal(0.0001, 0.01, size=(500, 20))

    pbo_result = ProbabilityOfBacktestOverfitting.compute_pbo(noise_returns, n_slices=6)
    assert "pbo" in pbo_result

    assert 0.20 <= pbo_result["pbo"] <= 0.80
