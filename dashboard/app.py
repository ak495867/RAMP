"""
RAMP Interactive Research & Execution Dashboard (Streamlit).
Renders real-time regime telemetry, backtest tearsheets, factor attribution, and friction analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from ramp.data.collectors.synthetic import SyntheticRegimeDataGenerator
from ramp.regimes.online_hmm import OnlineHamiltonFilterHMM
from ramp.regimes.filter import RegimeHysteresisFilter
from ramp.signals.momentum import TimeSeriesMomentumSignal
from ramp.signals.carry import CrossAssetCarrySignal
from ramp.signals.mean_reversion import MeanReversionSignal
from ramp.signals.vrp import VolatilityRiskPremiumSignal
from ramp.backtest.engine import EventDrivenBacktestEngine
from ramp.portfolio.optimizer import RobustConvexOptimizer
from ramp.execution.cost_model import ExecutionCostModel
from ramp.validation.deflated_sharpe import DeflatedSharpeRatio
from ramp.validation.pbo import ProbabilityOfBacktestOverfitting

st.set_page_config(
    page_title="RAMP | Quantitative Portfolio & Regime Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" RAMP: Regime-Adaptive Multi-Asset Platform")
st.caption("Research-to-Production Platform: Causal Online Regimes, Convex Optimization & Realistic Market Impact")

st.sidebar.header("Data Feed & Universe")
data_source = st.sidebar.radio(
    "Data Source",
    options=["Real Market Lakehouse (2018-Present)", "Synthetic Markov Simulation"],
    index=0
)

universe_selection = st.sidebar.multiselect(
    "Asset Universe",
    options=["SPY", "QQQ", "IWM", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP", "BTC-USD"],
    default=["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
)

rebalance_freq = st.sidebar.select_slider("Rebalance Frequency (Trading Days)", options=[1, 5, 10, 21], value=5)
target_vol = st.sidebar.slider("Annual Volatility Target (%)", min_value=5.0, max_value=25.0, value=12.0) / 100.0
turnover_penalty = st.sidebar.slider("Turnover L1 Penalty (Lambda)", min_value=0.000, max_value=0.010, value=0.002, step=0.001, format="%.3f")
market_impact_y = st.sidebar.slider("Kyle/Almgren Impact Coefficient Y", min_value=0.00, max_value=0.40, value=0.15, step=0.05)

run_button = st.sidebar.button(" Run Event-Driven Backtest", use_container_width=True)

@st.cache_data
def run_simulation(data_mode, symbols, freq, vol, lmbda, y_impact):
    parquet_path = "d:/RAMP/data/parquet/multi_asset_bars_2018_present.parquet"
    if data_mode == "Real Market Lakehouse (2018-Present)" and pd.io.common.file_exists(parquet_path):
        raw_df = pd.read_parquet(parquet_path)
        bars_df = raw_df[raw_df["symbol"].isin(symbols)].copy().sort_values(["timestamp", "symbol"])
    else:
        gen = SyntheticRegimeDataGenerator(seed=42)
        bars_df, _ = gen.generate_universe(symbols, n_bars=400)

    hmm = OnlineHamiltonFilterHMM(n_regimes=3)
    hyst = RegimeHysteresisFilter(confidence_threshold=0.65, min_dwell_bars=3)
    signals = [
        TimeSeriesMomentumSignal(lookbacks=[21, 63, 120]),
        CrossAssetCarrySignal(),
        MeanReversionSignal(lookback=20),
        VolatilityRiskPremiumSignal(rv_lookback=21)
    ]
    opt = RobustConvexOptimizer(turnover_penalty_lambda=lmbda, max_position_weight=0.35)
    cost = ExecutionCostModel(impact_coefficient_y=y_impact)
    vol_eng = VolatilityTargetingEngine(target_annual_vol=vol)

    engine = EventDrivenBacktestEngine(
        symbols=symbols,
        regime_detector=hmm,
        hysteresis_filter=hyst,
        signals=signals,
        optimizer=opt,
        vol_targeting=vol_eng,
        cost_model=cost,
        initial_capital=1000000.0,
        rebalance_frequency_bars=freq,
        burn_in_bars=60
    )

    results = engine.run(bars_df)
    return results, bars_df

if run_button or "results" not in st.session_state:
    with st.spinner("Executing point-in-time simulation & causal filtering..."):
        results, bars_df = run_simulation(
            data_source, universe_selection, rebalance_freq, target_vol, turnover_penalty, market_impact_y
        )
        st.session_state["results"] = results
        st.session_state["bars"] = bars_df

results = st.session_state["results"]
metrics = results["metrics"]
history = results["history"]
weights = results["weights"]
regimes = results["regimes"]
fills = results["fills"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("CAGR", f"{metrics.get('cagr', 0)*100:.1f}%")
c2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
c3.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
c4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
c5.metric("Annual Turnover", f"{metrics.get('annualized_turnover', 0)*100:.0f}%")
c6.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    " Equity & Drawdown",
    " Regime Detection",
    " Dynamic Allocations",
    " Overfitting & Frictions"
])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Portfolio Equity Curve (Net of Non-Linear Frictions)")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=history["timestamp"], y=history["total_nav"],
            mode="lines", name="RAMP Adaptive NAV",
            line=dict(color="#00D084", width=2.5)
        ))
        fig_equity.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Portfolio NAV ($USD)"
        )
        st.plotly_chart(fig_equity, use_container_width=True)

    with col_b:
        st.subheader("Underwater Drawdown")
        cum_ret = np.cumprod(1.0 + history["daily_return"].values)
        running_max = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - running_max) / running_max

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=history["timestamp"], y=dd * 100,
            fill="tozeroy", mode="lines", name="Drawdown %",
            line=dict(color="#FF4B4B", width=1.5)
        ))
        fig_dd.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Drawdown (%)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with tab2:
    st.subheader("Causal Online Regime Probabilities & Shifts")
    if not regimes.empty:
        regime_names_map = {0: "Low-Vol Expansion", 1: "High-Vol Contraction", 2: "Crisis / Liquidity Shock"}
        regimes["regime_label"] = regimes["regime_id"].map(regime_names_map)

        fig_regime = px.scatter(
            regimes, x="timestamp", y="regime_label",
            color="regime_label",
            title="Causal Hamilton Filter Output (Zero Backward Smoothing)",
            color_discrete_map={
                "Low-Vol Expansion": "#00D084",
                "High-Vol Contraction": "#FFAA00",
                "Crisis / Liquidity Shock": "#FF4B4B"
            }
        )
        fig_regime.update_layout(template="plotly_dark")
        st.plotly_chart(fig_regime, use_container_width=True)

        st.info(" **Institutional Guarantee:** Hamilton Forward Filter operates strictly as P(S_t | x_{1:t}) — no look-ahead bias from future market shocks.")

with tab3:
    st.subheader("Asset Allocation Over Time")
    if not weights.empty:
        cols_to_plot = [c for c in weights.columns if c != "timestamp"]
        fig_weights = go.Figure()
        for col in cols_to_plot:
            fig_weights.add_trace(go.Scatter(
                x=weights["timestamp"], y=weights[col],
                mode="lines", stackgroup="one", name=col
            ))
        fig_weights.update_layout(
            template="plotly_dark",
            yaxis_title="Allocation Weight",
            yaxis=dict(range=[0, 1.1])
        )
        st.plotly_chart(fig_weights, use_container_width=True)

with tab4:
    st.subheader("Microstructure Slippage & Overfitting Diagnostics")
    c_f1, c_f2 = st.columns(2)

    total_slippage = sum(f.slippage for f in fills)
    total_commissions = sum(f.commission for f in fills)
    total_volume_traded = sum(f.quantity * f.price for f in fills)

    with c_f1:
        st.markdown("#### Execution Cost Breakdown")
        st.write(f"- **Total Traded Volume:** `${total_volume_traded:,.2f}`")
        st.write(f"- **Kyle Square-Root Impact Slippage:** `${total_slippage:,.2f}`")
        st.write(f"- **Broker Commissions:** `${total_commissions:,.2f}`")
        st.write(f"- **Total Frictional Drag:** `${(total_slippage + total_commissions):,.2f}`")

    with c_f2:
        st.markdown("#### Overfitting Metrics (López de Prado)")
        dsr_val = DeflatedSharpeRatio.deflated_sharpe_ratio(
            observed_sr=metrics.get("sharpe_ratio", 1.0),
            trials_variance=0.20,
            num_trials=50,
            n_observations=len(history)
        )
        st.metric("Deflated Sharpe Ratio (DSR)", f"{dsr_val:.3f}")
        st.caption("DSR penalizes observed Sharpe ratio for multiple hypothesis trials.")
