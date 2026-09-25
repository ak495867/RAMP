"""
RAMP Head-to-Head Real-World Benchmark Evaluation.
Compares:
1. Static 60/40 Equity/Treasury (SPY/TLT)
2. Static Risk Parity / Inverse Volatility (All Assets)
3. RAMP Regime-Adaptive Multi-Asset Platform

All evaluated over 2018-present real market history under identical non-linear execution frictions.
"""

import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ramp.backtest.benchmarks import BenchmarkEvaluator
from ramp.backtest.engine import EventDrivenBacktestEngine
from ramp.execution.cost_model import ExecutionCostModel
from ramp.portfolio.optimizer import RobustConvexOptimizer
from ramp.portfolio.vol_target import VolatilityTargetingEngine
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.signals.mean_reversion import MeanReversionSignal
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.vrp import VolatilityRiskPremiumSignal
from ramp.validation.deflated_sharpe import DeflatedSharpeRatio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PARQUET_FILE = Path("d:/RAMP/data/parquet/multi_asset_bars_2018_present.parquet")

def main():
    print("=" * 80)
    print("[*] RAMP REAL-WORLD STRATEGY BENCHMARK EVALUATION (2018 - PRESENT)")
    print("=" * 80)

    if not PARQUET_FILE.exists():
        print(f"Error: {PARQUET_FILE} not found. Please run ingest_real_data.py first.")
        return

    bars_df = pd.read_parquet(PARQUET_FILE)
    print(f"[*] Loaded {len(bars_df):,} real-world market bars from Lakehouse.")
    print(f"    Date Range: {bars_df['timestamp'].min().strftime('%Y-%m-%d')} to {bars_df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"    Instruments: {sorted(bars_df['symbol'].unique())}\n")

    tradeable_symbols = ["SPY", "QQQ", "IWM", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "BTC-USD"]

    available_symbols = [s for s in tradeable_symbols if s in bars_df["symbol"].unique()]
    print(f"[*] Tradeable Universe ({len(available_symbols)} assets): {available_symbols}")

    cost_model = ExecutionCostModel(
        impact_coefficient_y=0.15,
        default_half_spread_bps=2.0,
        commission_per_share=0.005,
        max_adv_participation=0.05
    )

    print("\n[1/3] Simulating Benchmark 1: Static 60/40 Equity/Treasury (SPY/TLT)...")
    res_60_40 = BenchmarkEvaluator.run_static_60_40(
        bars_df=bars_df,
        equity_sym="SPY",
        bond_sym="TLT",
        rebalance_freq_bars=21,
        cost_model=cost_model
    )

    print("[2/3] Simulating Benchmark 2: Static Risk Parity across Universe...")
    res_risk_parity = BenchmarkEvaluator.run_static_risk_parity(
        bars_df=bars_df,
        symbols=available_symbols,
        rebalance_freq_bars=21,
        lookback_vol=63,
        cost_model=cost_model
    )

    print("[3/3] Simulating RAMP: Regime-Adaptive Multi-Asset Platform...")
    hmm = OnlineHamiltonFilterHMM(n_regimes=3)
    hyst = RegimeHysteresisFilter(confidence_threshold=0.65, min_dwell_bars=3)
    signals = [
        TimeSeriesMomentumSignal(lookbacks=[21, 63, 120]),
        CrossAssetCarrySignal(),
        MeanReversionSignal(lookback=20),
        VolatilityRiskPremiumSignal(rv_lookback=21)
    ]
    optimizer = RobustConvexOptimizer(
        turnover_penalty_lambda=0.002,
        max_position_weight=0.30,
        max_asset_class_weight=0.55
    )
    vol_target = VolatilityTargetingEngine(target_annual_vol=0.12)

    engine = EventDrivenBacktestEngine(
        symbols=available_symbols,
        regime_detector=hmm,
        hysteresis_filter=hyst,
        signals=signals,
        optimizer=optimizer,
        vol_targeting=vol_target,
        cost_model=cost_model,
        rebalance_frequency_bars=5,                             
        burn_in_bars=60
    )

    res_ramp = engine.run(bars_df)

    m_6040 = res_60_40["metrics"]
    m_rp = res_risk_parity["metrics"]
    m_ramp = res_ramp["metrics"]

    slip_6040 = sum(f.slippage for f in res_60_40["fills"])
    comm_6040 = sum(f.commission for f in res_60_40["fills"])

    slip_rp = sum(f.slippage for f in res_risk_parity["fills"])
    comm_rp = sum(f.commission for f in res_risk_parity["fills"])

    slip_ramp = sum(f.slippage for f in res_ramp["fills"])
    comm_ramp = sum(f.commission for f in res_ramp["fills"])

    print("\n" + "=" * 80)
    print("🏆 HEAD-TO-HEAD INSTITUTIONAL SCOREBOARD (2018 - PRESENT)")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Static 60/40':>14} | {'Risk Parity':>14} | {'RAMP Adaptive':>14}")
    print("-" * 80)
    print(f"{'CAGR':<28} | {m_6040.get('cagr', 0)*100:>13.2f}% | {m_rp.get('cagr', 0)*100:>13.2f}% | {m_ramp.get('cagr', 0)*100:>13.2f}%")
    print(f"{'Annualized Volatility':<28} | {m_6040.get('annualized_volatility', 0)*100:>13.2f}% | {m_rp.get('annualized_volatility', 0)*100:>13.2f}% | {m_ramp.get('annualized_volatility', 0)*100:>13.2f}%")
    print(f"{'Sharpe Ratio':<28} | {m_6040.get('sharpe_ratio', 0):>14.2f} | {m_rp.get('sharpe_ratio', 0):>14.2f} | {m_ramp.get('sharpe_ratio', 0):>14.2f}")
    print(f"{'Sortino Ratio':<28} | {m_6040.get('sortino_ratio', 0):>14.2f} | {m_rp.get('sortino_ratio', 0):>14.2f} | {m_ramp.get('sortino_ratio', 0):>14.2f}")
    print(f"{'Max Drawdown':<28} | {m_6040.get('max_drawdown', 0)*100:>13.2f}% | {m_rp.get('max_drawdown', 0)*100:>13.2f}% | {m_ramp.get('max_drawdown', 0)*100:>13.2f}%")
    print(f"{'Calmar Ratio':<28} | {m_6040.get('calmar_ratio', 0):>14.2f} | {m_rp.get('calmar_ratio', 0):>14.2f} | {m_ramp.get('calmar_ratio', 0):>14.2f}")
    print(f"{'Daily VaR (95%)':<28} | {m_6040.get('var_95_daily', 0)*100:>13.2f}% | {m_rp.get('var_95_daily', 0)*100:>13.2f}% | {m_ramp.get('var_95_daily', 0)*100:>13.2f}%")
    print(f"{'Annual Turnover':<28} | {m_6040.get('annualized_turnover', 0)*100:>13.1f}% | {m_rp.get('annualized_turnover', 0)*100:>13.1f}% | {m_ramp.get('annualized_turnover', 0)*100:>13.1f}%")
    print(f"{'Total Trades Executed':<28} | {len(res_60_40['fills']):>14d} | {len(res_risk_parity['fills']):>14d} | {len(res_ramp['fills']):>14d}")
    print(f"{'Total Frictional Drag ($)':<28} | ${slip_6040 + comm_6040:>13,.2f} | ${slip_rp + comm_rp:>13,.2f} | ${slip_ramp + comm_ramp:>13,.2f}")
    print("=" * 80)

    dsr_ramp = DeflatedSharpeRatio.deflated_sharpe_ratio(
        observed_sr=m_ramp.get("sharpe_ratio", 1.0),
        trials_variance=0.15,
        num_trials=20,
        n_observations=len(res_ramp["history"])
    )
    print(f"\n[*] López de Prado Deflated Sharpe Ratio (DSR): {dsr_ramp:.4f}")
    if dsr_ramp >= 0.95:
        print("    -> STATISTICALLY SIGNIFICANT OUTPERFORMANCE (p > 0.95) AFTER SELECTION BIAS PENALTY.")
    else:
        print("    -> Caution: Sharpe ratio requires larger sample or lower parameter search variance.")

    print("\n[OK] Head-to-head real-world evaluation complete.\n")

if __name__ == "__main__":
    main()
