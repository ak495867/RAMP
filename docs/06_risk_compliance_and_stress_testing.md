# 6. Pre-Trade Compliance, EVT & Black-Swan Stress Testing

---

## 6.1 Pre-Trade Compliance Gatekeeper

Before any order is dispatched to a broker, it must pass through [`ramp/execution/compliance.py`](file:///d:/RAMP/ramp/execution/compliance.py).

```
   Target Orders
         |
         v
+------------------+     Fail
| ADV Cap (<= 5%)  | -----------> Clamp Order Size
+------------------+
         | Pass
         v
+------------------+     Fail
| Single Cap (15%) | -----------> Clamp Position
+------------------+
         | Pass
         v
+------------------+     Fail
| Silo / Leverage  | -----------> Reject / Scale Down
+------------------+
         | Pass
         v
+------------------+     Tripped
| Kill Switch (-5%)| -----------> HALT ALL REBALANCING & LIQUIDATE RISK
+------------------+
         | Pass
         v
   Approved Execution
```

### 1. ADV Volume Limit
$$\text{Order Shares } Q_i \le 0.05 \times \text{ADV}_{20, i}$$

### 2. Single-Name Concentration Limit
$$|w_i| \le 0.15 \quad (\text{Max } 15\% \text{ in any single instrument})$$

### 3. Asset-Class Silo Constraints
$$\sum_{i \in \text{Equities}} w_i \le 0.55, \quad \sum_{i \in \text{Crypto}} w_i \le 0.15, \quad \sum_{i \in \text{Commodities}} w_i \le 0.35, \quad \sum_{i \in \text{Fixed Income}} w_i \le 0.70$$

### 4. Leverage Buffers (Reg-T / Basel III)
$$\text{Gross Leverage} = \sum_{i=1}^N |w_i| \le 1.50\text{x}, \quad \text{Net Exposure} = |\sum_{i=1}^N w_i| \le 1.00\text{x}$$

### 5. Automated Circuit Breaker Kill-Switch
If rolling 5-day drawdown breaches **$-5.00\%$**, the kill-switch triggers:
* All pending buy orders are immediately cancelled.
* High-beta exposures are liquidated into cash and short-term Treasuries (`SHY`).
* Rebalancing is locked and emergency alerts are dispatched.

---

## 6.2 Extreme Value Theory (EVT) & Peaks-Over-Threshold

Standard Gaussian distribution models fail in liquidity panics because normal curves severely underestimate fat tails. Under the **Pickands–Balkema–de Haan Theorem**, losses exceeding a high threshold $u$ converge asymptotically to a **Generalized Pareto Distribution (GPD)**:

$$G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

In [`ramp/validation/evt_copula.py`](file:///d:/RAMP/ramp/validation/evt_copula.py):
* Shape parameter $\xi$ measures tail heaviness ($\xi > 0$ indicates fat power-law Fréchet tails).
* Scale parameter $\beta$ measures dispersion beyond threshold $u$.

### EVT Value-at-Risk (VaR 99%)
$$\text{VaR}_\alpha = u + \frac{\beta}{\xi} \left[ \left(\frac{N}{N_u} (1 - \alpha)\right)^{-\xi} - 1 \right]$$

### EVT Expected Shortfall / Conditional VaR (CVaR 99%)
$$\text{ES}_\alpha = \frac{\text{VaR}_\alpha}{1 - \xi} + \frac{\beta - \xi u}{1 - \xi}$$

---

## 6.3 Student-t Copula Multivariate Stress Simulator

Gaussian copulas suffer from zero lower tail dependence:
$$\lambda_L = \lim_{q \to 0} P(U_1 \le q \mid U_2 \le q) = 0$$
In a crisis, Gaussian models predict that diversification remains intact. In reality, asset correlations spike to 1.0.

RAMP implements a **Student-$t$ Copula** with low degrees of freedom ($\nu = 4$), generating non-zero lower tail dependence:

$$\lambda_L = 2 t_{\nu + 1}\left(-\sqrt{\frac{(\nu + 1)(1 - \rho)}{1 + \rho}}\right) > 0$$

### Simulation Protocol:
1. Simulates $M = 5,000$ synthetic cross-asset crisis scenarios using joint Student-$t$ dependency structures.
2. Inverts uniform marginals through individual asset empirical distributions with GPD fat tails.
3. Stress-tests the target portfolio against simultaneous systemic drawdowns (e.g., US equities down $35\%$, oil down $50\%$, crypto down $60\%$, Treasury yields inverted).
4. Verifies that the portfolio's worst-case drawdown and EVT CVaR remain within fund risk mandates.
