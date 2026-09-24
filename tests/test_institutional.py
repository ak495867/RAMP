from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from ramp.portfolio.covariance import HighDimensionalCovarianceEstimator
from ramp.portfolio.factor_risk import BarraFactorRiskModel
from ramp.regimes.macro_quadrant import MacroQuadrantClassifier, YieldCurvePCA
from ramp.execution.compliance import PreTradeComplianceEngine
from ramp.execution.almgren_chriss import AlmgrenChrissExecutionOptimizer
from ramp.core.types import Order, OrderSide, OrderType


def test_marchenko_pastur_rmt_denoising():
    rng = np.random.default_rng(42)
    n_assets = 20
    n_obs = 100
    noise_returns = rng.normal(0, 1, size=(n_obs, n_assets))
    sample_cov = np.cov(noise_returns, rowvar=False)

    denoised_cov = HighDimensionalCovarianceEstimator.denoise_covariance(sample_cov, n_obs)

    assert denoised_cov.shape == (n_assets, n_assets)
    evals, _ = np.linalg.eigh(denoised_cov)
    assert np.all(evals >= -1e-6)

    shrunk_cov, intensity = HighDimensionalCovarianceEstimator.ledoit_wolf_shrinkage(noise_returns)
    assert shrunk_cov.shape == (n_assets, n_assets)
    assert 0.0 <= intensity <= 1.0


def test_barra_factor_risk_decomposition():
    rng = np.random.default_rng(101)
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    
    asset_rets = pd.DataFrame(
        rng.normal(0.0005, 0.015, size=(150, 5)),
        index=dates,
        columns=["SPY", "QQQ", "TLT", "GLD", "BTC"]
    )
    factor_rets = pd.DataFrame(
        rng.normal(0.0004, 0.012, size=(150, 3)),
        index=dates,
        columns=["market_beta", "rates_duration", "crypto_liquidity"]
    )

    model = BarraFactorRiskModel()
    X, F, Delta = model.estimate_factor_model(asset_rets, factor_rets)

    assert X.shape == (5, 3)
    assert F.shape == (3, 3)
    assert Delta.shape == (5, 5)

    weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
    decomp = model.decompose_portfolio_risk(weights, X, F, Delta)

    assert "total_annual_vol" in decomp
    assert "systematic_risk_pct" in decomp
    assert "specific_risk_pct" in decomp
    assert pytest.approx(decomp["systematic_risk_pct"] + decomp["specific_risk_pct"], abs=0.5) == 100.0


def test_macro_quadrant_and_yield_curve_pca():
    classifier = MacroQuadrantClassifier()
    features = pd.DataFrame({
        "growth_indicator": [0.02, 0.01, -0.01, -0.02],
        "inflation_indicator": [0.03, -0.01, 0.02, -0.01]
    })
    classifier.fit(features)

    state = classifier.filter_step(np.array([0.05, -0.02]), datetime(2026, 1, 1))
    assert state.regime_id == 0

    state_stagflation = classifier.filter_step(np.array([-0.05, 0.04]), datetime(2026, 1, 2))
    assert state_stagflation.regime_id == 2

    yc_data = pd.DataFrame({
        "2Y": [0.04, 0.042, 0.039, 0.045],
        "5Y": [0.042, 0.043, 0.041, 0.046],
        "10Y": [0.045, 0.046, 0.044, 0.048],
        "30Y": [0.048, 0.049, 0.047, 0.050]
    })
    pca = YieldCurvePCA()
    pca.fit(yc_data)
    transformed = pca.transform(np.array([0.041, 0.043, 0.046, 0.049]))
    assert "level" in transformed
    assert "slope" in transformed


def test_pre_trade_compliance_engine():
    compliance = PreTradeComplianceEngine(
        max_adv_participation=0.05,
        max_single_weight=0.15,
        drawdown_kill_switch_pct=-0.05
    )

    orders = [
        Order(order_id="1", symbol="SPY", side=OrderSide.BUY, quantity=1000),
        Order(order_id="2", symbol="BTC", side=OrderSide.BUY, quantity=50000)
    ]
    adv_map = {"SPY": 100000, "BTC": 20000}
    price_map = {"SPY": 500.0, "BTC": 60000.0}

    res = compliance.validate_orders(
        orders=orders,
        current_nav=1000000.0,
        adv_map=adv_map,
        price_map=price_map,
        current_drawdown=-0.01
    )

    assert len(res.clamped_orders) == 2
    assert res.clamped_orders[1].quantity <= 1000

    kill_res = compliance.validate_orders(
        orders=orders,
        current_nav=1000000.0,
        adv_map=adv_map,
        price_map=price_map,
        current_drawdown=-0.06
    )
    assert not kill_res.is_compliant
    assert kill_res.kill_switch_active
    assert len(kill_res.clamped_orders) == 0


def test_almgren_chriss_execution():
    optimizer = AlmgrenChrissExecutionOptimizer()
    schedule = optimizer.compute_optimal_trajectory(
        total_shares=10000.0,
        time_horizon_hours=6.5,
        n_steps=10,
        daily_volatility=0.015,
        mid_price=100.0
    )

    assert len(schedule) == 10
    total_traded = sum(item["shares_to_trade"] for item in schedule)
    assert pytest.approx(total_traded, abs=1.0) == 10000.0
    assert schedule[-1]["remaining_shares"] <= 0.001
