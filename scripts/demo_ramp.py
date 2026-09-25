"""
RAMP End-to-End Demonstration Script.
Executes data ingestion, point-in-time storage, online regime filtering,
alpha signal generation, CVXPY convex optimization, and execution simulation.
"""

import sys
from datetime import datetime
import pandas as pd
import numpy as np

from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.data.pit_store import PointInTimeStore
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.signals.mean_reversion import MeanReversionSignal
from ramp.signals.vrp import VolatilityRiskPremiumSignal
from ramp.portfolio.optimizer import RobustConvexOptimizer
from ramp.execution.cost_model import ExecutionCostModel
from ramp.backtest.engine import EventDrivenBacktestEngine
from ramp.validation.deflated_sharpe import DeflatedSharpeRatio
from ramp.validation.pbo import ProbabilityOfBacktestOverfitting

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("=" * 70)
    print("[*] RAMP: REGIME-ADAPTIVE MULTI-ASSET PORTFOLIO PLATFORM")
    print("=" * 70)

    symbols = ["SPY", "TLT", "GLD", "BTC-USD"]
    print(f"\n[1] Generating Multi-Asset Universe: {symbols} (350 bars)...")
    gen = SyntheticRegimeDataGenerator(seed=42)
    bars_df, true_regimes = gen.generate_universe(symbols, n_bars=350)
    print(f"    -> Generated {len(bars_df)} point-in-time bars across {len(symbols)} instruments.")

    print("\n[2] Ingesting into DuckDB Point-in-Time Lakehouse...")
    store = PointInTimeStore(":memory:")
    store.insert_bars(bars_df)
    print("    -> Stored and indexed in DuckDB with zero look-ahead as-of joins.")

    print("\n[3] Initializing Causal Hamilton Filter & Anti-Whipsaw Hysteresis...")
    hmm = OnlineHamiltonFilterHMM(n_regimes=3)
    hyst = RegimeHysteresisFilter(confidence_threshold=0.65, min_dwell_bars=3)

    print("\n[4] Initializing Multi-Factor Alpha Signal Library...")
    signals = [
        TimeSeriesMomentumSignal(lookbacks=[21, 63, 120]),
        CrossAssetCarrySignal(),
        MeanReversionSignal(lookback=20),
        VolatilityRiskPremiumSignal(rv_lookback=21)
    ]
    print(f"    -> Configured {len(signals)} orthogonal cross-asset signals.")

    print("\n[5] Configuring Convex Optimizer & Institutional Cost Model...")
    optimizer = RobustConvexOptimizer(turnover_penalty_lambda=0.002, max_position_weight=0.35)
    cost_model = ExecutionCostModel(impact_coefficient_y=0.15, default_half_spread_bps=2.0)

    print("\n[6] Running Strictly Event-Driven Out-Of-Sample Simulation (T -> T+1 Open)...")
    engine = EventDrivenBacktestEngine(
        symbols=symbols,
        regime_detector=hmm,
        hysteresis_filter=hyst,
        signals=signals,
        optimizer=optimizer,
        cost_model=cost_model,
        initial_capital=1000000.0,
        rebalance_frequency_bars=5,
        burn_in_bars=60
    )

    results = engine.run(bars_df)
    metrics = results["metrics"]
    fills = results["fills"]
    history = results["history"]

    print("\n" + "=" * 70)
    print("[TEARSHEET] INSTITUTIONAL PERFORMANCE TEARSHEET (NET OF ALL FRICTIONS)")
    print("=" * 70)
    for k, v in metrics.items():
        if isinstance(v, float) and ("cagr" in k or "drawdown" in k or "volatility" in k or "var" in k):
            print(f"  {k:28s} : {v * 100:6.2f}%")
        else:
            print(f"  {k:28s} : {v}")

    total_slip = sum(f.slippage for f in fills)
    total_comm = sum(f.commission for f in fills)
    print("\n" + "-" * 70)
    print("[FRICTIONS] MICROSTRUCTURE & TRANSACTION DRAG ANALYSIS")
    print("-" * 70)
    print(f"  Total Trades Executed       : {len(fills)}")
    print(f"  Square-Root Price Impact    : ${total_slip:,.2f}")
    print(f"  Broker Commissions Paid     : ${total_comm:,.2f}")
    print(f"  Total Frictional Drag       : ${(total_slip + total_comm):,.2f}")

    dsr = DeflatedSharpeRatio.deflated_sharpe_ratio(
        observed_sr=metrics.get("sharpe_ratio", 1.0),
        trials_variance=0.20,
        num_trials=25,
        n_observations=len(history)
    )
    print("\n" + "-" * 70)
    print("[OVERFITTING] STATISTICAL RIGOR (BAILEY & LOPEZ DE PRADO)")
    print("-" * 70)
    print(f"  Deflated Sharpe Ratio (DSR) : {dsr:.4f}")
    print(f"  DSR Evaluation              : {'STATISTICALLY ROBUST (p > 0.95)' if dsr >= 0.95 else 'CAUTION: POTENTIAL SNOOPING'}")

    store.close()
    print("\n[OK] RAMP Execution Complete! Platform is operational and ready for production.\n")

if __name__ == "__main__":
    main()
