from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from ramp.paper.reconciler import PortfolioDriftReconciler
from ramp.paper.alpaca_broker import AlpacaBrokerGateway
from ramp.execution.obi_slicer import OrderBookImbalanceSlicer
from ramp.signals.cot_positioning import COTPositioningSignal
from ramp.regimes.hsmm import HiddenSemiMarkovModel
from ramp.signals.factor_pruning import DynamicFactorPruner
from ramp.core.alerts import MultiChannelAlertManager, AlertSeverity
from ramp.core.types import Order, OrderSide, OrderType, Position

def test_portfolio_drift_reconciler():
    reconciler = PortfolioDriftReconciler(drift_tolerance_pct=0.02, min_trade_dollar=50.0)
    current_positions = {
        "SPY": Position(symbol="SPY", quantity=1000, avg_price=500.0, current_price=500.0),
        "TLT": Position(symbol="TLT", quantity=1000, avg_price=100.0, current_price=100.0)
    }
    nav = 1000000.0
    prices = {"SPY": 500.0, "TLT": 100.0, "GLD": 200.0}

    target_weights = {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20}
    orders, drift_map, needed = reconciler.calculate_rebalance_orders(nav, current_positions, target_weights, prices)

    assert needed
    assert drift_map["SPY"] == 0.0
    assert drift_map["TLT"] == 0.20
    assert drift_map["GLD"] == 0.20
    assert len(orders) == 2

def test_alpaca_broker_gateway():
    gateway = AlpacaBrokerGateway(is_paper=True)
    assert gateway.get_nav() == 1000000.0

    order = Order(order_id="t1", symbol="SPY", side=OrderSide.BUY, quantity=100, limit_price=500.0)
    fill = gateway.submit_order(order)

    assert fill.quantity == 100
    assert fill.price == 500.0
    positions = gateway.get_positions()
    assert "SPY" in positions
    assert positions["SPY"].quantity == 100

def test_order_book_imbalance_slicer():
    slicer = OrderBookImbalanceSlicer(strong_imbalance_threshold=0.30)
    decision_buy_bullish = slicer.evaluate_microstructure_action(
        side=OrderSide.BUY,
        bid_volume=50000,
        ask_volume=10000,
        bid_price=500.00,
        ask_price=500.05
    )
    assert decision_buy_bullish.action == "AGGRESSIVE_BID_CROSS"
    assert decision_buy_bullish.urgency_multiplier > 1.0

    decision_buy_bearish = slicer.evaluate_microstructure_action(
        side=OrderSide.BUY,
        bid_volume=10000,
        ask_volume=50000,
        bid_price=500.00,
        ask_price=500.05
    )
    assert decision_buy_bearish.action == "PASSIVE_WAIT_PULLBACK"
    assert decision_buy_bearish.urgency_multiplier < 1.0

def test_cot_positioning_signal():
    signal = COTPositioningSignal(crowding_threshold_z=2.0)
    dates = pd.date_range("2024-01-01", periods=60, freq="W")
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": "SPY",
        "close": np.linspace(400, 500, 60),
        "cot_net_speculative": np.linspace(1000, 50000, 60)
    })
    views = signal.generate_views(df, dates[-1], ["SPY"])
    assert "SPY" in views
    assert views["SPY"].expected_return < 0.0

def test_hsmm_regime_filtering():
    hsmm = HiddenSemiMarkovModel()
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, 100)
    hsmm.fit(features)

    state = hsmm.filter_step(np.array([2.5]), datetime(2026, 1, 1))
    assert state.regime_id in [0, 1, 2]
    assert 0.0 <= sum(state.probabilities.values()) <= 1.0001

def test_dynamic_factor_pruner():
    pruner = DynamicFactorPruner(min_ir_threshold=0.20)
    dates = pd.date_range("2024-01-01", periods=50, freq="B")

    signals = pd.DataFrame({
        "good_factor": np.linspace(1, 50, 50),
        "decayed_factor": np.random.default_rng(42).normal(0, 1, 50)
    }, index=dates)

    fwd_returns = pd.DataFrame({
        "good_factor": np.linspace(0.01, 0.50, 50),
        "decayed_factor": np.random.default_rng(101).normal(0, 0.02, 50)
    }, index=dates)

    health = pruner.evaluate_factor_health(signals, fwd_returns)
    assert health["good_factor"]["is_active"]

    weights = {"good_factor": 0.5, "decayed_factor": 0.5}
    pruned = pruner.compute_pruned_factor_weights(weights, health)
    assert pruned["good_factor"] > pruned["decayed_factor"]

def test_multi_channel_alert_manager():
    alert_mgr = MultiChannelAlertManager()
    payload = alert_mgr.format_alert_payload(
        title="Circuit Breaker Activated",
        message="Drawdown exceeded 5%",
        severity=AlertSeverity.CRITICAL,
        metadata={"drawdown_pct": -5.2, "action": "HALT_REBALANCE"}
    )
    assert "[CRITICAL]" in payload["title"]
    assert payload["metadata"]["drawdown_pct"] == -5.2
