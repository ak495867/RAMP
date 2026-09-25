"""
Unit tests for data validation, continuous futures roll, and DuckDB PIT store.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest
from ramp.data.validator import MarketDataValidator, DataValidationError
from ramp.data.rolls import ContinuousFuturesBuilder
from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.data.pit_store import PointInTimeStore

def test_validator_rejects_negative_prices():
    bad_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
        "open": [100.0, -5.0],
        "high": [105.0, 100.0],
        "low": [98.0, 95.0],
        "close": [102.0, 99.0],
        "volume": [1000, 2000],
    })
    is_valid, errors = MarketDataValidator.validate_bars(bad_df, "TEST")
    assert not is_valid
    assert any("non-positive" in e for e in errors)

def test_validator_rejects_inverted_high_low():
    bad_df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2026-01-01")],
        "open": [100.0],
        "high": [90.0],              
        "low": [95.0],
        "close": [92.0],
        "volume": [1000],
    })
    is_valid, errors = MarketDataValidator.validate_bars(bad_df, "TEST")
    assert not is_valid
    assert any("High < Low" in e for e in errors)

def test_futures_roll_yield():

    ry_back = ContinuousFuturesBuilder.calculate_roll_yield(front_price=105.0, next_price=100.0, days_to_expiry=30)
    assert ry_back > 0.0

    ry_cont = ContinuousFuturesBuilder.calculate_roll_yield(front_price=95.0, next_price=100.0, days_to_expiry=30)
    assert ry_cont < 0.0

def test_synthetic_data_generation_and_pit_store():
    gen = SyntheticRegimeDataGenerator(seed=42)
    bars, regimes = gen.generate_universe(["SPY", "TLT", "GLD"], n_bars=100)

    assert len(bars) > 0
    assert len(regimes) > 0

    store = PointInTimeStore(":memory:")
    store.insert_bars(bars)

    pit_bars = store.get_bars_pit(
        symbols=["SPY"],
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 2, 1)
    )
    assert not pit_bars.empty
    assert (pit_bars["symbol"] == "SPY").all()

    features = store.get_causal_regime_features(
        symbols=["SPY", "TLT", "GLD"],
        as_of_date=datetime(2020, 3, 1),
        lookback_bars=20
    )
    assert not features.empty
    assert "basket_return" in features.columns
    store.close()
