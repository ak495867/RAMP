import numpy as np
import pytest

from ramp.validation.evt_copula import ExtremeValueTheoryEngine, CopulaStressSimulator
from ramp.execution.volume_profile import IntradayVolumeProfiler, UShapedVWAPSlicer


def test_extreme_value_theory_gpd_fitting():
    rng = np.random.default_rng(42)
    normal_losses = rng.normal(0.001, 0.015, size=500)
    pareto_tails = rng.pareto(a=3.0, size=50) * 0.03
    losses = np.concatenate([normal_losses, pareto_tails])

    engine = ExtremeValueTheoryEngine(tail_quantile=0.95)
    u, xi, beta = engine.fit_gpd_tail(losses)

    assert u > 0.0
    assert beta > 0.0

    metrics = engine.compute_evt_risk_metrics(losses, confidence_level=0.99)
    assert "evt_var" in metrics
    assert "evt_cvar" in metrics
    assert metrics["evt_cvar"] >= metrics["evt_var"]


def test_copula_stress_simulator():
    corr = np.array([
        [1.0, 0.7, 0.4],
        [0.7, 1.0, 0.3],
        [0.4, 0.3, 1.0]
    ])
    vols = np.array([0.015, 0.020, 0.035])
    weights = np.array([0.5, 0.3, 0.2])

    simulator = CopulaStressSimulator(degrees_of_freedom=4, random_state=123)
    scenarios = simulator.simulate_t_copula_shocks(corr, vols, n_scenarios=1000)

    assert scenarios.shape == (1000, 3)

    stress_eval = simulator.evaluate_portfolio_stress(weights, scenarios)
    assert "worst_case_drawdown" in stress_eval
    assert "evt_var_99" in stress_eval
    assert "evt_cvar_99" in stress_eval
    assert stress_eval["worst_case_drawdown"] > 0.0


def test_intraday_volume_profiler_u_shape():
    profiler = IntradayVolumeProfiler()
    profile = profiler.profile

    assert len(profile) == 13
    assert pytest.approx(float(np.sum(profile)), abs=1e-5) == 1.0

    assert profile[0] > profile[5]
    assert profile[-1] > profile[5]

    cum_profile = profiler.get_cumulative_profile()
    assert len(cum_profile) == 13
    assert np.all(np.diff(cum_profile) > 0)
    assert pytest.approx(float(cum_profile[-1]), abs=1e-5) == 1.0


def test_u_shaped_vwap_slicer():
    slicer = UShapedVWAPSlicer()
    total_order = 50000.0
    adv = 2000000.0

    schedule = slicer.generate_vwap_schedule(
        total_shares=total_order,
        adv_20=adv,
        urgency_decay=0.04
    )

    assert len(schedule) == 13
    total_scheduled = sum(item["scheduled_shares"] for item in schedule)
    assert pytest.approx(total_scheduled, abs=1.0) == total_order
    assert schedule[-1]["remaining_shares"] <= 0.01

    assert schedule[0]["scheduled_shares"] > schedule[5]["scheduled_shares"]
