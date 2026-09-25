# 5. Microstructure, Slippage & Optimal Execution

---

## 5.1 The Frictionless Backtest Fallacy

In academic backtests, strategies appear profitable because they assume trades fill at the exact closing print with infinite liquidity and zero market impact. In real-world institutional execution:
1. Large orders move the limit order book against the trader.
2. Orders submitted at the close cannot execute until market open of the next day, incurring overnight gap risk.
3. Midday trading incurs wider bid-ask spreads and lower market depth.

---

## 5.2 Institutional Transaction Cost & Market Impact Model

In [`ramp/execution/cost_model.py`](../ramp/execution/cost_model.py), RAMP calculates execution fills using the non-linear **Kyle / Almgren-Chriss Square-Root Law**:

$$P_{\text{exec}} = P_{\text{mid}} \left(1 \pm \frac{s}{2} \pm Y \cdot \sigma_{\text{daily}} \sqrt{\frac{Q}{\text{ADV}}} \pm \eta_{\text{delay}}\right)$$

Where:
* $Q$: Order quantity in shares/contracts.
* $\text{ADV}$: 20-day Average Daily Volume.
* $s / 2$: Half-spread crossing cost (e.g. 1.5 to 2.5 bps).
* $\sigma_{\text{daily}}$: Realized daily volatility.
* $Y$: Market impact coefficient ($Y \approx 0.15$).
* $\eta_{\text{delay}}$: Overnight execution latency slippage ($T$ Close $\to T+1$ Open).

### ADV Participation Limit
No single order is permitted to exceed $5\%$ of daily volume ($Q \le 0.05 \cdot \text{ADV}$). Orders exceeding this limit are partially filled or sliced across multiple sessions.

---

## 5.3 Intraday U-Shaped Volume Profiler

In [`ramp/execution/volume_profile.py`](../ramp/execution/volume_profile.py), RAMP models the empirical **U-shaped volume distribution** across the 390 minutes of the US trading session:

```
Volume %
  20% |  ***                                               ***
      |  *  *                                             *  *
  10% |  *   *                                           *   *
      |  *    *    *                               *    *    *
   4% |  *     **** ******************************* ****     *
      +-------------------------------------------------------
        09:30       11:00          13:00         15:00     16:00
```

### Calibrated 30-Minute Bin Fractions
* **09:30 - 10:00 (Open Auction & Overnight Digestion)**: $18\%$ of ADV
* **10:00 - 10:30**: $12\%$ of ADV
* **10:30 - 11:30**: $8\%$ to $6\%$ of ADV
* **11:30 - 13:30 (Midday Lunch Lull)**: $4\%$ to $5\%$ of ADV (low liquidity trough)
* **13:30 - 15:00**: $6\%$ to $8\%$ of ADV
* **15:00 - 15:30**: $10\%$ of ADV
* **15:30 - 16:00 (Market-on-Close & Index Rebalance)**: $17\%$ of ADV

---

## 5.4 U-Shaped Volume-Clock VWAP Slicing

[`UShapedVWAPSlicer`](../ramp/execution/volume_profile.py#L33-L77) transforms Almgren-Chriss liquidation urgency from chronological clock time to **volume-clock time**:

$$n_k = \frac{w_{\text{vol}, k} \cdot e^{-\lambda t_k}}{\sum_{j=1}^{13} w_{\text{vol}, j} \cdot e^{-\lambda t_j}} \times Q_{\text{total}}$$

Child order sizes dynamically scale with expected market depth: trading aggressively during high-liquidity open/close windows and throttling order flow during illiquid midday hours.

---

## 5.5 Order Book Imbalance (OBI) Tactical Router

In [`ramp/execution/obi_slicer.py`](../ramp/execution/obi_slicer.py), child orders receive tactical routing instructions based on top-of-book depth:

$$\text{OBI}_t = \frac{V_{\text{bid}, t} - V_{\text{ask}, t}}{V_{\text{bid}, t} + V_{\text{ask}, t}} \in [-1, 1]$$

### Tactical Microstructure Rules:
* **Buy Order & $\text{OBI} \ge +0.30$** (Strong bid support): Route aggressive limit orders at the bid to capture liquidity before the ask marks up.
* **Buy Order & $\text{OBI} \le -0.30$** (Heavy ask selling cascade): Route passive limit orders below the bid and throttle execution pace to capture price concessions.
* **Sell Order & $\text{OBI} \le -0.30$**: Route aggressive limit orders at the ask.
* **Sell Order & $\text{OBI} \ge +0.30$**: Route passive limit orders above the ask.
