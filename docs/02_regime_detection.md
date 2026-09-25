# 2. Causal Online Regime Detection Engine

---

## 2.1 The Offline Smoothing Scam

Standard quantitative papers frequently apply offline hidden Markov models using the Baum-Welch backward pass:
$$P(S_t = j \mid x_1, x_2, \dots, x_T)$$
This backward variable $\beta_t(i) = P(x_{t+1:T} \mid S_t = i)$ explicitly evaluates observations from the future ($t+1 \dots T$). In a backtest run across 2015 to 2024, an offline model classifying December 2019 "knows" that the March 2020 COVID crash occurred, creating severe in-sample look-ahead bias.

---

## 2.2 Online Hamilton Forward Filter

In [`ramp/regimes/online_hmm.py`](../ramp/regimes/online_hmm.py), RAMP strictly enforces the **Hamilton Forward Algorithm**, calculating state probabilities using only past and present observations:

$$P(S_t = j \mid x_{1:t}) = \frac{f(x_t \mid S_t = j) \sum_{i=1}^K P(S_t = j \mid S_{t-1} = i) P(S_{t-1} = i \mid x_{1:t-1})}{\sum_{k=1}^K f(x_t \mid S_t = k) \sum_{i=1}^K P(S_t = k \mid S_{t-1} = i) P(S_{t-1} = i \mid x_{1:t-1})}$$

Where:
* $f(x_t \mid S_t = j)$ is the multivariate Gaussian emission density:
  $$f(x_t \mid S_t = j) = \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2} (x_t - \mu_j)^T \Sigma_j^{-1} (x_t - \mu_j)\right)$$
* $A_{ij} = P(S_t = j \mid S_{t-1} = i)$ is the row-stochastic transition probability matrix.

### Preventing Label Switching via Variance Canonicalization
Across re-estimation windows, hidden Markov models can suffer from arbitrary label switching (state 0 swapping meaning with state 1). RAMP permanently solves this by canonicalizing states ordered by the trace of their emission covariance matrix:
$$\text{Tr}(\Sigma_0) \le \text{Tr}(\Sigma_1) \le \dots \le \text{Tr}(\Sigma_{K-1})$$
* **Regime 0**: Low-Volatility Expansion (bull trend, normal carry, low dispersion).
* **Regime 1**: High-Volatility Contraction (defensive posture, elevated dispersion).
* **Regime 2**: Crisis Liquidity Shock (extreme correlation spikes, flight to cash).

---

## 2.3 Bayesian Online Change-Point Detection (BOCPD)

In [`ramp/regimes/bocpd.py`](../ramp/regimes/bocpd.py), RAMP implements Adams & MacKay (2007) Bayesian Change-Point Detection. 

BOCPD maintains a recursive posterior distribution over the **run-length** $r_t$ (the number of time steps since the last structural changepoint):

$$P(r_t \mid x_{1:t}) \propto P(r_t, x_{1:t})$$

### Recursive Run-Length Update
$$P(r_t = r_{t-1} + 1, x_{1:t}) = P(r_{t-1}, x_{1:t-1}) \cdot P(x_t \mid r_{t-1}, x_t^{(r)}) \cdot (1 - H(r_{t-1}))$$
$$P(r_t = 0, x_{1:t}) = \sum_{r_{t-1}} P(r_{t-1}, x_{1:t-1}) \cdot P(x_t \mid r_{t-1}, x_t^{(r)}) \cdot H(r_{t-1})$$

Where $H(r) = \frac{1}{\lambda}$ is a constant hazard function. When a structural shock hits, probability mass immediately collapses to $r_t = 0$, signaling an instantaneous regime break without lag.

---

## 2.4 Hidden Semi-Markov Models (HSMM) with Explicit Duration

In [`ramp/regimes/hsmm.py`](../ramp/regimes/hsmm.py), RAMP resolves the memoryless limitation of standard HMMs (which enforce geometric dwell times $P(d) = (1 - a_{ii}) a_{ii}^{d-1}$). 

The HSMM explicitly parameterizes state dwell probability distributions $p_i(d)$ (e.g. Poisson or Negative Binomial):
* Regime 0 (Low-Vol Expansion): expected duration $\mu_0 \approx 200$ trading days.
* Regime 1 (High-Vol Contraction): expected duration $\mu_1 \approx 60$ trading days.
* Regime 2 (Crisis Shock): expected duration $\mu_2 \approx 15$ trading days.

The transition probability incorporates the elapsed dwell time $d_t$:
$$P(\text{Exit State } i \mid d_t) = \text{CDF}_{\text{Poisson}}(d_t; \mu_i)$$
This prevents premature regime flips during early stages of long-term bull markets.

---

## 2.5 Bridgewater 4-Quadrant Macro Engine & Yield Curve PCA

In [`ramp/regimes/macro_quadrant.py`](../ramp/regimes/macro_quadrant.py), macroeconomic momentum is classified across Growth and Inflation axes:

| Regime Quadrant | Economic Conditions | Favorable Asset Classes |
|---|---|---|
| **Quadrant 0** | Rising Growth, Falling Inflation | US Tech / Growth Equities, High-Beta Beta |
| **Quadrant 1** | Rising Growth, Rising Inflation | Raw Commodities, Energy, Real Estate, TIPS |
| **Quadrant 2** | Falling Growth, Rising Inflation | Gold, Cash Reserves, Short Duration |
| **Quadrant 3** | Falling Growth, Falling Inflation | Long-Term Treasuries, Sovereign Duration |

### Yield Curve Principal Component Analysis (PCA)
Decomposes 2Y, 5Y, 10Y, and 30Y Treasury yields into three orthogonal components:
1. **Level ($\beta_1$)**: Parallel shifts across the entire yield curve (monetary policy rate cycle).
2. **Slope ($\beta_2$)**: Steepening vs. inversion (economic growth and recession expectations).
3. **Curvature ($\beta_3$)**: Belly butterfly shift (term premium and liquidity dislocations).

---

## 2.6 Anti-Whipsaw Hysteresis Filter

In [`ramp/regimes/filter.py`](../ramp/regimes/filter.py), raw regime probabilities pass through state-persistence constraints before triggering portfolio adjustments:

$$S_t^* = \begin{cases} 
k^* & \text{if } P(S_t = k^* \mid x_{1:t}) \ge \tau_{\text{enter}} \text{ and } \text{Dwell}_{S_{t-1}} > D_{\min} \\
S_{t-1}^* & \text{otherwise}
\end{cases}$$

Where:
* Confidence threshold $\tau_{\text{enter}} = 0.65$ (requires $65\%$ probability mass to switch).
* Minimum dwell constraint $D_{\min} = 3$ trading days (suppresses single-day noise).
* **Crisis Fast-Trigger Bypass**: If candidate regime is Crisis ($k^* = 2$) and probability exceeds $0.55$, the dwell constraint is bypassed to protect capital.
