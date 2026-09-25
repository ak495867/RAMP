"""
End-to-End Event-Driven Backtest & Execution System Tests.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.backtest.engine import EventDrivenBacktestEngine
from ramp.portfolio.optimizer import RobustConvexOptimizer
from ramp.execution.cost_model import ExecutionCostModel

def test_full_event_driven_backtest():
    symbols = ["SPY", "TLT", "GLD", "BTC-USD"]
    gen = SyntheticRegimeDataGenerator(seed=101)
    bars, true_regimes = gen.generate_universe(symbols, n_bars=200)

    hmm = OnlineHamiltonFilterHMM(n_regimes=3)
    hyst = RegimeHysteresisFilter(confidence_threshold=0.60, min_dwell_bars=3)
    signals = [
        TimeSeriesMomentumSignal(lookbacks=[21, 63]),
        CrossAssetCarrySignal()
    ]
    opt = RobustConvexOptimizer(turnover_penalty_lambda=0.001)
    cost = ExecutionCostModel(impact_coefficient_y=0.10)

    engine = EventDrivenBacktestEngine(
        symbols=symbols,
        regime_detector=hmm,
        hysteresis_filter=hyst,
        signals=signals,
        optimizer=opt,
        cost_model=cost,
        initial_capital=1000000.0,
        rebalance_frequency_bars=5,
        burn_in_bars=60
    )

    results = engine.run(bars)

    assert "metrics" in results
    assert "history" in results
    assert "weights" in results
    assert "fills" in results

    metrics = results["metrics"]
    assert "cagr" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert metrics["max_drawdown"] <= 0.0

    history = results["history"]
    assert not history.empty
    assert history["total_nav"].iloc[-1] > 0.0

    fills = results["fills"]
    assert len(fills) > 0
    total_slip = sum(f.slippage for f in fills)
    total_comm = sum(f.commission for f in fills)
    assert total_slip > 0.0
    assert total_comm > 0.0
