"""
Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR).
Formulated by Marcos López de Prado and David Bailey (2014).
Penalizes the observed Sharpe ratio for multiple hypothesis testing, skewness, kurtosis, and track record length.
"""

from typing import Optional
import numpy as np
import scipy.stats as stats


class DeflatedSharpeRatio:
    """
    Computes PSR and DSR to test if an observed Sharpe ratio is statistically significant
    after accounting for data snooping over N trials.
    """

    @staticmethod
    def probabilistic_sharpe_ratio(
        observed_sr: float,
        benchmark_sr: float,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculates Probabilistic Sharpe Ratio (PSR):
        PSR(SR*) = Z( (SR - SR*) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1)/4 * SR^2) )
        """
        if n_observations <= 2:
            return 0.5

        sr = observed_sr
        sr_bm = benchmark_sr
        T = n_observations

        denom = np.sqrt(max(1.0 - skewness * sr + ((kurtosis - 1.0) / 4.0) * (sr ** 2), 1e-6))
        z_stat = (sr - sr_bm) * np.sqrt(T - 1.0) / denom
        psr = float(stats.norm.cdf(z_stat))
        return psr

    @classmethod
    def deflated_sharpe_ratio(
        cls,
        observed_sr: float,
        trials_variance: float,
        num_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0
    ) -> float:
        """
        Calculates Deflated Sharpe Ratio (DSR).
        The benchmark Sharpe ratio is the expected maximum Sharpe ratio under the null hypothesis:
        E[max(SR_0)] = sqrt(trials_variance) * ((1 - Euler_gamma) * Z^-1(1 - 1/N) + Euler_gamma * Z^-1(1 - 1/(N*e)))
        """
        if num_trials <= 1:
            return cls.probabilistic_sharpe_ratio(observed_sr, 0.0, n_observations, skewness, kurtosis)

        # Euler-Mascheroni constant
        euler = 0.5772156649
        std_trials = np.sqrt(max(trials_variance, 1e-6))

        # Expected maximum Sharpe under null of independent random strategies
        z1 = stats.norm.ppf(1.0 - 1.0 / num_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (num_trials * np.e))
        expected_max_sr = std_trials * ((1.0 - euler) * z1 + euler * z2)

        dsr = cls.probabilistic_sharpe_ratio(
            observed_sr=observed_sr,
            benchmark_sr=expected_max_sr,
            n_observations=n_observations,
            skewness=skewness,
            kurtosis=kurtosis
        )
        return float(dsr)
