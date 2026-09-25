# 4. High-Dimensional Risk & Convex Optimization

---

## 4.1 The Curse of Dimensionality ($N/T$ Collapse)

When expanding the asset universe from 10 to 40+ instruments, the sample covariance matrix $\hat{\Sigma}$ becomes mathematically ill-conditioned. If the number of assets $N$ approaches sample size $T$ ($N / T \to 1$), standard covariance inversion in Markowitz quadratic optimization inverts eigenvalue noise. The optimizer places large leveraged bets on spurious negative correlations that do not exist out of sample.

---

## 4.2 Random Matrix Theory (RMT) Marchenko-Pastur Denoising

In [`ramp/portfolio/covariance.py`](../ramp/portfolio/covariance.py), RAMP filters empirical covariance matrices using the **Marchenko-Pastur Theorem**.

### Theoretical Noise Bounds
Under the null hypothesis of pure uncorrelated noise, the eigenvalues of a random correlation matrix are bounded by:
$$\lambda_{\min}^{\max} = \sigma^2 \left(1 \pm \sqrt{\frac{N}{T}}\right)^2$$

### Denoising Algorithm:
1. Decompose sample correlation matrix: $C = V \Lambda V^T$.
2. Identify true factor eigenvalues: $\lambda_i > \lambda_{\max}$.
3. Replace pure noise eigenvalues ($\lambda_i \le \lambda_{\max}$) with their average trace value:
   $$\tilde{\lambda}_i = \frac{1}{N - M} \sum_{j=1}^{N - M} \lambda_j \quad \forall \lambda_i \le \lambda_{\max}$$
4. Reconstruct clean, invertible correlation matrix $\tilde{C} = V \tilde{\Lambda} V^T$.
5. Re-scale to covariance: $\tilde{\Sigma} = \text{diag}(\sigma) \tilde{C} \text{diag}(\sigma)$.

---

## 4.3 Barra-Style Multi-Factor Risk Decomposition

In [`ramp/portfolio/factor_risk.py`](../ramp/portfolio/factor_risk.py), asset returns are decomposed into systematic risk factors and idiosyncratic specific risk:

$$r_i = \sum_{k=1}^K X_{ik} f_k + \epsilon_i$$

Total portfolio variance is decomposed into two orthogonal components:

$$\sigma_P^2 = \underbrace{w^T \left(X F X^T\right) w}_{\text{Systematic Factor Risk}} + \underbrace{w^T \Delta w}_{\text{Specific Idiosyncratic Risk}}$$

Where:
* $X \in \mathbb{R}^{N \times K}$ is the factor exposure matrix (Market Beta, Rates Duration, Credit Spread, Commodity Inflation, Momentum, Value, Crypto Liquidity).
* $F \in \mathbb{R}^{K \times K}$ is the factor covariance matrix.
* $\Delta \in \mathbb{R}^{N \times N}$ is the diagonal specific risk matrix.

---

## 4.4 Regime-Conditioned Black-Litterman Model

In [`ramp/portfolio/black_litterman.py`](../ramp/portfolio/black_litterman.py), equilibrium market priors and active alpha views are blended conditioned on the active regime $S_t$:

### 1. Regime Prior Equilibrium
$$\Pi_k = \delta_k \tilde{\Sigma} w_{\text{mkt}}$$
Where risk aversion parameter $\delta_k$ dynamically adjusts:
* Low-Vol Expansion: $\delta_0 = 2.5$
* High-Vol Contraction: $\delta_1 = 4.5$
* Crisis Liquidity Shock: $\delta_2 = 7.5$

### 2. Posterior Expected Return & Covariance
$$\mu_{\text{post}} = \left[(\tau \tilde{\Sigma})^{-1} + P^T \Omega^{-1} P\right]^{-1} \left[(\tau \tilde{\Sigma})^{-1} \Pi_k + P^T \Omega^{-1} Q\right]$$
$$V_{\text{post}} = \tilde{\Sigma} + \left[(\tau \tilde{\Sigma})^{-1} + P^T \Omega^{-1} P\right]^{-1}$$

---

## 4.5 CVXPY Convex Optimization with L1 Turnover Penalty

In [`ramp/portfolio/optimizer.py`](../ramp/portfolio/optimizer.py), target weights $w \in \mathbb{R}^N$ solve the convex problem:

$$\min_{w} \quad \frac{\gamma}{2} w^T V_{\text{post}} w - \mu_{\text{post}}^T w + \lambda_{\text{turnover}} \|w - w_{\text{prev}}\|_1$$

Subject to:
$$\sum_{i=1}^N w_i \le 1.0 \quad (\text{Cash buffer allowed})$$
$$0.0 \le w_i \le 0.15 \quad (\text{Single-name concentration cap})$$
$$\sum_{j \in \text{AssetClass}_m} w_j \le \text{Cap}_m \quad (\text{Asset-class silo limits})$$

### Hierarchical Risk Parity (HRP) Fallback
If the convex solver encounters numerical instability, RAMP automatically falls back to **Hierarchical Risk Parity** ([`ramp/portfolio/hrp.py`](../ramp/portfolio/hrp.py)), performing tree clustering on correlation distance matrices without matrix inversion.

---

## 4.6 Dynamic Realized Volatility Targeting

In [`ramp/portfolio/vol_target.py`](../ramp/portfolio/vol_target.py), portfolio weights are continuously scaled to maintain a constant annual risk profile (e.g. $\sigma_{\text{target}} = 12\%$):

$$\text{Scale} = \min\left(1.20, \frac{\sigma_{\text{target}}}{\sqrt{252 \cdot w^T V_{\text{post}} w}}\right)$$
$$w_{\text{final}} = \text{Scale} \cdot w$$
$$w_{\text{cash}} = 1.0 - \sum_{i=1}^N w_{\text{final}, i}$$

During crisis regimes, as asset volatilities spike, the engine automatically de-leverages into unallocated cash earning the risk-free rate.
