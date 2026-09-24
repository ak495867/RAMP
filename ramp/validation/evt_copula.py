from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.stats as stats


class ExtremeValueTheoryEngine:

    def __init__(self, tail_quantile: float = 0.95):
        self.tail_quantile = tail_quantile

    def fit_gpd_tail(self, losses: np.ndarray) -> Tuple[float, float, float]:
        pos_losses = np.sort(losses[losses > 0])
        if len(pos_losses) < 20:
            threshold = float(np.percentile(losses, self.tail_quantile * 100))
            return threshold, 0.1, float(np.std(losses))

        threshold = float(np.percentile(pos_losses, self.tail_quantile * 100))
        exceedances = pos_losses[pos_losses > threshold] - threshold
        if len(exceedances) < 5:
            return threshold, 0.1, float(np.std(pos_losses))

        c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
        xi = float(c)
        beta = float(scale)
        return threshold, xi, beta

    def compute_evt_risk_metrics(
        self,
        losses: np.ndarray,
        confidence_level: float = 0.99
    ) -> Dict[str, float]:
        n_total = len(losses)
        u, xi, beta = self.fit_gpd_tail(losses)
        n_u = np.sum(losses > u)

        if n_u == 0 or beta <= 0:
            var_empirical = float(np.percentile(losses, confidence_level * 100))
            return {
                "threshold_u": round(float(u), 4),
                "tail_index_xi": 0.1,
                "scale_beta": 0.01,
                "evt_var": round(var_empirical, 4),
                "evt_cvar": round(var_empirical * 1.25, 4)
            }

        p_exceed = n_u / n_total
        alpha = confidence_level

        if xi != 0:
            term = ((1.0 - alpha) / p_exceed) ** (-xi)
            var_evt = u + (beta / xi) * (term - 1.0)
            if xi < 1.0:
                cvar_evt = (var_evt / (1.0 - xi)) + ((beta - xi * u) / (1.0 - xi))
            else:
                cvar_evt = var_evt * 1.5
        else:
            var_evt = u - beta * np.log((1.0 - alpha) / p_exceed)
            cvar_evt = var_evt + beta

        return {
            "threshold_u": round(float(u), 4),
            "tail_index_xi": round(float(xi), 4),
            "scale_beta": round(float(beta), 4),
            "evt_var": round(float(var_evt), 4),
            "evt_cvar": round(float(cvar_evt), 4)
        }


class CopulaStressSimulator:

    def __init__(self, degrees_of_freedom: int = 4, random_state: int = 42):
        self.df = degrees_of_freedom
        self.rng = np.random.default_rng(random_state)

    def simulate_t_copula_shocks(
        self,
        corr_matrix: np.ndarray,
        asset_volatilities: np.ndarray,
        asset_means: Optional[np.ndarray] = None,
        n_scenarios: int = 5000
    ) -> np.ndarray:
        n_assets = corr_matrix.shape[0]
        if asset_means is None:
            mu = np.zeros(n_assets)
        else:
            mu = np.asarray(asset_means).reshape(-1)

        evals, evecs = np.linalg.eigh(corr_matrix)
        evals = np.maximum(evals, 1e-8)
        clean_corr = evecs @ np.diag(evals) @ evecs.T
        diag_inv = np.diag(1.0 / np.sqrt(np.diag(clean_corr)))
        clean_corr = diag_inv @ clean_corr @ diag_inv

        L = np.linalg.cholesky(clean_corr)
        Z = self.rng.standard_normal(size=(n_scenarios, n_assets))
        correlated_normals = Z @ L.T

        chi2_samples = self.rng.chisquare(df=self.df, size=n_scenarios) / self.df
        chi2_samples = np.sqrt(chi2_samples).reshape(-1, 1)

        t_variates = correlated_normals / chi2_samples

        uniform_margins = stats.t.cdf(t_variates, df=self.df)

        simulated_returns = np.zeros_like(uniform_margins)
        for i in range(n_assets):
            vol_i = asset_volatilities[i]
            simulated_returns[:, i] = stats.t.ppf(
                np.clip(uniform_margins[:, i], 1e-5, 1.0 - 1e-5),
                df=5,
                loc=mu[i],
                scale=vol_i
            )

        return simulated_returns

    def evaluate_portfolio_stress(
        self,
        weights: np.ndarray,
        simulated_scenarios: np.ndarray
    ) -> Dict[str, float]:
        w = np.asarray(weights).reshape(-1)
        port_losses = -(simulated_scenarios @ w)

        evt_engine = ExtremeValueTheoryEngine(tail_quantile=0.95)
        evt_metrics = evt_engine.compute_evt_risk_metrics(port_losses, confidence_level=0.99)

        max_loss = float(np.max(port_losses))
        median_loss = float(np.median(port_losses))

        return {
            "worst_case_drawdown": round(max_loss * 100.0, 2),
            "median_scenario_loss": round(median_loss * 100.0, 2),
            "evt_var_99": round(evt_metrics["evt_var"] * 100.0, 2),
            "evt_cvar_99": round(evt_metrics["evt_cvar"] * 100.0, 2),
            "tail_index_xi": evt_metrics["tail_index_xi"],
            "scale_beta": evt_metrics["scale_beta"]
        }
