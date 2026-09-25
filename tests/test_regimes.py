"""
Unit tests for causal online regime detection: Hamilton HMM, BOCPD, and Hysteresis.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest
from ramp.core.types import RegimeState
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.bocpd import BayesianOnlineChangePointDetector
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator

@pytest.fixture
def synthetic_data():
    gen = SyntheticRegimeDataGenerator(seed=123)
    bars, regimes = gen.generate_universe(["SPY", "TLT", "GLD"], n_bars=300)
    pivoted = bars.pivot(index="timestamp", columns="symbol", values="close")
    returns = pivoted.pct_change().dropna()
    features = pd.DataFrame({
        "basket_return": returns.mean(axis=1),
        "volatility": returns.std(axis=1) * np.sqrt(252),
    })
    return features

def test_online_hamilton_filter_causality(synthetic_data):
    features = synthetic_data
    burn_in = features.iloc[:150]
    out_of_sample = features.iloc[150:]

    hmm = OnlineHamiltonFilterHMM(n_regimes=3)
    hmm.fit(burn_in)

    assert hmm.is_fitted

    variances = [np.trace(cov) for cov in hmm.covariances]
    assert variances[0] <= variances[1] <= variances[2]

    states = []
    for dt, row in out_of_sample.iterrows():
        state = hmm.filter_step(row.values, dt)
        assert isinstance(state, RegimeState)
        probs = list(state.probabilities.values())
        assert pytest.approx(sum(probs), abs=1e-5) == 1.0
        assert 0 <= state.regime_id < 3
        states.append(state)

    assert len(states) == len(out_of_sample)

def test_bocpd_changepoint_detection():
    bocpd = BayesianOnlineChangePointDetector(hazard_rate=0.02)

    rng = np.random.default_rng(42)
    stable_period = rng.normal(0.001, 0.01, size=100)
    shock_period = rng.normal(-0.08, 0.05, size=10)
    series = np.concatenate([stable_period, shock_period])

    bocpd.fit(stable_period[:30])

    dt = datetime(2026, 1, 1)
    states = []
    for val in series:
        dt += timedelta(days=1)
        state = bocpd.filter_step(np.array([val]), dt)
        states.append(state)

    shock_states = states[100:]
    max_break_prob = max(s.probabilities[1] for s in shock_states)
    assert max_break_prob > 0.40

def test_hysteresis_filter_prevents_whipsaw():
    hyst = RegimeHysteresisFilter(confidence_threshold=0.70, min_dwell_bars=3)
    dt = datetime(2026, 1, 1)

    s1 = RegimeState(dt, 0, "low_vol", {0: 0.90, 1: 0.10, 2: 0.0}, False, 0.2)
    out1 = hyst.filter(s1)
    assert out1.regime_id == 0

    dt += timedelta(days=1)
    s2 = RegimeState(dt, 1, "high_vol", {0: 0.45, 1: 0.55, 2: 0.0}, False, 0.6)
    out2 = hyst.filter(s2)
    assert out2.regime_id == 0                      

    dt += timedelta(days=1)
    s3 = RegimeState(dt, 1, "high_vol", {0: 0.15, 1: 0.85, 2: 0.0}, False, 0.4)
    out3 = hyst.filter(s3)
    assert out3.regime_id == 0                                             

    dt += timedelta(days=1)
    s4 = RegimeState(dt, 1, "high_vol", {0: 0.10, 1: 0.90, 2: 0.0}, False, 0.3)
    out4 = hyst.filter(s4)
    assert out4.regime_id == 1
    assert out4.is_transition is True
