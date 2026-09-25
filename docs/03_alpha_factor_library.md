# 3. Alpha Factor Library & Dynamic Pruning

---

## 3.1 Overview & Signal Interface

Every alpha signal in RAMP inherits from [`BaseSignal`](file:///d:/RAMP/ramp/signals/base.py) and produces standardized [`SignalView`](file:///d:/RAMP/ramp/core/types.py#L90-L98) instances:
* `expected_return`: Annualized expected return ($\mu$).
* `confidence`: Bayesian certainty parameter $\in [0.05, 1.0]$.
* `horizon_bars`: Expected holding period horizon in trading days.

---

## 3.2 Signal Catalog

### 1. Multi-Horizon Time-Series Momentum (TSMOM)
* **File**: [`ramp/signals/momentum.py`](file:///d:/RAMP/ramp/signals/momentum.py)
* **Economic Logic**: Capital flows and institutional slow-moving capital create persistent multi-month price trends (Moskowitz, Ooi, Pedersen 2012).
* **Mathematical Formulation**:
  Combines 1M (21d), 3M (63d), and 12M (252d) lookback returns scaled by ex-ante annualized volatility:
  $$z_{i, L} = \text{clip}\left(\frac{P_{i, t} - P_{i, t-L}}{P_{i, t-L} \cdot \sigma_{i, t} \sqrt{L / 252}}, -3.0, 3.0\right)$$
  $$\mu_i = \left(\frac{\sigma_{\text{target}}}{\sigma_{i, t}}\right) \cdot \left(\sum_k w_k z_{i, L_k}\right) \times 0.05$$
* **Expected Holding Period**: 21 to 63 days.
* **Failure Condition**: Sharp momentum turning points / V-shaped market reversals.

---

### 2. Cross-Asset Carry
* **File**: [`ramp/signals/carry.py`](file:///d:/RAMP/ramp/signals/carry.py)
* **Economic Logic**: Investors demand compensation for holding risk assets assuming static spot prices (roll yields, yield curves, interest rate differentials).
* **Components**:
  * *Commodities*: Futures term structure basis (backwardation vs contango roll yield).
  * *Fixed Income*: 10Y minus 2Y Treasury yield curve slope.
  * *Currencies*: Uncovered interest rate parity differentials.
* **Expected Holding Period**: 63 to 252 days.
* **Failure Condition**: Sudden carry unwinds (e.g. JPY carry trade squeezes).

---

### 3. Ornstein-Uhlenbeck Mean Reversion
* **File**: [`ramp/signals/mean_reversion.py`](file:///d:/RAMP/ramp/signals/mean_reversion.py)
* **Economic Logic**: Short-term liquidity dislocations cause asset prices to deviate from local exponential moving averages.
* **Mathematical Formulation**:
  $$z_t = \frac{P_t - \mu_{\text{EMA}, 20}}{\sigma_{\text{rolling}, 20}}$$
  $$\mu_{\text{expected}} = -z_t \times 0.05$$
* **Regime Conditioning**: During **Crisis** regimes ($S_t = 2$), mean reversion views are discounted by $70\%$ to prevent catching falling knives.
* **Expected Holding Period**: 5 to 20 days.

---

### 4. Volatility Risk Premium (VRP)
* **File**: [`ramp/signals/vrp.py`](file:///d:/RAMP/ramp/signals/vrp.py)
* **Economic Logic**: Options market participants consistently overpay for downside catastrophe insurance, creating a structural spread between Implied Volatility ($IV$) and Realized Volatility ($RV$).
* **Mathematical Formulation**:
  $$\text{VRP}_t = IV_{\text{VIX}, t} - RV_{\text{21d}, t}$$
  * When $\text{VRP} > +2.0\%$: Healthy market, harvest equity risk premium.
  * When $\text{VRP} < -2.0\%$: Volatility spike / panic, long defensive duration and gold.

---

### 5. CFTC Commitments of Traders (COT) Positioning
* **File**: [`ramp/signals/cot_positioning.py`](file:///d:/RAMP/ramp/signals/cot_positioning.py)
* **Economic Logic**: Commercial hedgers are informed fundamental participants, while leveraged speculators represent trend-following liquidity. When speculative crowding reaches extreme limits, upside momentum exhausts.
* **Mathematical Formulation**:
  $$Z_{\text{COT}} = \frac{\text{Net Speculative Position}_t - \mu_{52}}{\sigma_{52}}$$
  * When $Z_{\text{COT}} \ge +2.0$: Crowded long trade, contrarian short view.
  * When $Z_{\text{COT}} \le -2.0$: Speculative wash-out, short squeeze long view.

---

## 3.3 AutoML Dynamic Factor Pruning

In [`ramp/signals/factor_pruning.py`](file:///d:/RAMP/ramp/signals/factor_pruning.py), factors that suffer from structural decay (factor crowding) are automatically identified and pruned:

### Information Coefficient (IC) Tracking
Calculates rolling 63-day Spearman rank correlation between factor forecasts and realized 21-day forward returns:
$$\text{IC}_t = \text{SpearmanCorr}\left(\text{Factor Score}_t, \text{Forward Returns}_{t+1:t+21}\right)$$

### Information Ratio (IR) Hurdle
$$\text{IR} = \frac{\text{Mean}(\text{IC})}{\text{Std}(\text{IC})}$$
* If $\text{IR} \ge 0.20$ and $\text{Mean}(\text{IC}) > 0$: Factor is healthy, weight is scaled by IR.
* If $\text{IR} < 0.20$ or $\text{Mean}(\text{IC}) \le 0$: Factor is pruned (weight set to $0.0$) until predictive power recovers.
