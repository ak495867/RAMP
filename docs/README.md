# RAMP Technical Documentation & Architectural Blueprint

Welcome to the comprehensive engineering and mathematical documentation for **RAMP (Regime-Adaptive Multi-Asset Portfolio Research & Execution Platform)**.

---

##  Documentation Index

1. [Data Infrastructure & Point-in-Time Lakehouse](file:///d:/RAMP/docs/01_data_infrastructure.md)
   - DuckDB OLAP architecture, continuous futures roll stitching, point-in-time as-of joins, and eliminating look-ahead bias.
2. [Causal Online Regime Detection Engine](file:///d:/RAMP/docs/02_regime_detection.md)
   - Hamilton forward filtering, BOCPD run-length posteriors, HSMM explicit duration distributions, Bridgewater 4-quadrant macro engine, and anti-whipsaw hysteresis.
3. [Alpha Factor Library & Dynamic Pruning](file:///d:/RAMP/docs/03_alpha_factor_library.md)
   - Mathematical formulation and economic intuition of TSMOM, Cross-Asset Carry, OU Mean-Reversion, VRP, CFTC COT positioning, and rolling Information Coefficient (IC) pruning.
4. [High-Dimensional Risk & Convex Optimization](file:///d:/RAMP/docs/04_high_dimensional_risk_and_optimization.md)
   - Random Matrix Theory (RMT) Marchenko-Pastur denoising, Ledoit-Wolf shrinkage, Barra risk decomposition, Regime Black-Litterman, and CVXPY L1 turnover penalization.
5. [Microstructure, Slippage & Optimal Execution](file:///d:/RAMP/docs/05_microstructure_and_execution.md)
   - Kyle / Almgren-Chriss square-root price impact, intraday U-shaped volume profiling, volume-clock VWAP slicing, and Order Book Imbalance (OBI) queue routing.
6. [Pre-Trade Compliance, EVT & Black-Swan Stress Testing](file:///d:/RAMP/docs/06_risk_compliance_and_stress_testing.md)
   - Pre-trade safety gatekeeper, drawdown kill-switches, Peaks-Over-Threshold Extreme Value Theory (EVT) GPD tail modeling, and Student-t copula multivariate stress simulator.
7. [Live Multi-Broker OMS/EMS & Cloud Operations](file:///d:/RAMP/docs/07_live_oms_ems_and_deployment.md)
   - Portfolio drift reconciler, Alpaca/IBKR gateway connectivity, multi-channel webhook alerting (Discord/Slack/Telegram), and Docker Compose orchestration.

---

##  System Philosophy: The Non-Negotiable Rules

```
                      +-----------------------------------+
                      |      RAMP System Philosophy       |
                      +-----------------------------------+
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
+-------v-------+               +-------v-------+               +-------v-------+
|  Strict Point |               | Non-Linear    |               | Combinatorial |
|  -in-Time     |               | Microstructure|               | Cross-        |
|  Causality    |               | Friction      |               | Validation    |
+---------------+               +---------------+               +---------------+
```

1. **Strict Point-in-Time Causality**:
   No indicator calculated on day $t$ close is permitted to observe information released after market close of day $t$. Fills are strictly modeled at day $t+1$ market open.
2. **Online Filtering over Offline Smoothing**:
   All regime probabilities must be calculated via forward filtering $P(S_t \mid x_{1:t})$ without backward passes ($P(S_t \mid x_{1:T})$).
3. **Non-Linear Microstructure Drag**:
   Orders must absorb half-spread crossing, Kyle square-root market impact, broker commissions, and ADV participation limits ($\le 5\%$ ADV).
4. **Combinatorial Statistical Rigor**:
   All parameter sets must be evaluated using Combinatorial Purged Cross-Validation (CPCV) and Bailey & López de Prado Deflated Sharpe Ratios (DSR) to penalize selection bias.
