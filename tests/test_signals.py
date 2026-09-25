"""
Unit tests for RAMP alpha signals: TSMOM, Carry, Mean Reversion, VRP.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.signals.mean_reversion import MeanReversionSignal
from ramp.signals.vrp import VolatilityRiskPremiumSignal

@pytest.fixture
def market_history():
    gen = SyntheticRegimeDataGenerator(seed=42)
    bars, _ = gen.generate_universe(["SPY", "TLT", "GLD", "BTC-USD"], n_bars=300)
    return bars

def test_tsmom_signal(market_history):
    as_of = market_history["timestamp"].max()
    symbols = ["SPY", "TLT", "GLD", "BTC-USD"]
    tsmom = TimeSeriesMomentumSignal(lookbacks=[21, 63, 120])
    views = tsmom.generate_views(market_history, as_of, symbols)

    assert len(views) == len(symbols)
    for sym in symbols:
        view = views[sym]
        assert not np.isnan(view.expected_return)
        assert 0.0 < view.confidence <= 1.0
        assert -0.40 <= view.expected_return <= 0.40

def test_carry_signal(market_history):
    as_of = market_history["timestamp"].max()
    symbols = ["SPY", "TLT", "GLD"]
    carry = CrossAssetCarrySignal()
    views = carry.generate_views(market_history, as_of, symbols)

    assert len(views) == len(symbols)
    for sym in symbols:
        view = views[sym]
        assert not np.isnan(view.expected_return)
        assert 0.0 <= view.confidence <= 1.0

def test_mean_reversion_signal(market_history):
    as_of = market_history["timestamp"].max()
    symbols = ["SPY", "TLT"]
    mr = MeanReversionSignal(lookback=20)
    views = mr.generate_views(market_history, as_of, symbols)

    assert len(views) == len(symbols)
    for sym in symbols:
        view = views[sym]
        assert not np.isnan(view.expected_return)
        assert abs(view.expected_return) <= 0.25

def test_vrp_signal(market_history):
    as_of = market_history["timestamp"].max()
    symbols = ["SPY", "TLT"]
    vrp = VolatilityRiskPremiumSignal(rv_lookback=21)
    views = vrp.generate_views(market_history, as_of, symbols)

    assert len(views) == len(symbols)
    for sym in symbols:
        view = views[sym]
        assert not np.isnan(view.expected_return)
        assert view.confidence > 0.0
