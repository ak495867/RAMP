from typing import Tuple
import numpy as np
from sklearn.covariance import LedoitWolf

class HighDimensionalCovarianceEstimator:

    @staticmethod
    def marchenko_pastur_bounds(n_assets: int, n_observations: int, variance: float = 1.0) -> Tuple[float, float]:
        q = n_assets / max(n_observations, 1)
        lambda_min = variance * (1.0 - np.sqrt(q)) ** 2
        lambda_max = variance * (1.0 + np.sqrt(q)) ** 2
        return float(lambda_min), float(lambda_max)

    @classmethod
    def denoise_correlation_rmt(cls, corr_matrix: np.ndarray, n_observations: int) -> np.ndarray:
        n_assets = corr_matrix.shape[0]
        if n_assets <= 2:
            return corr_matrix

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        _, lambda_max = cls.marchenko_pastur_bounds(n_assets, n_observations, variance=1.0)

        clean_evals = eigenvalues.copy()
        noise_mask = clean_evals <= lambda_max
        if np.any(noise_mask):
            mean_noise_eval = np.mean(clean_evals[noise_mask])
            clean_evals[noise_mask] = mean_noise_eval

        clean_corr = eigenvectors @ np.diag(clean_evals) @ eigenvectors.T
        diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(clean_corr), 1e-8)))
        clean_corr = diag_inv_sqrt @ clean_corr @ diag_inv_sqrt
        np.fill_diagonal(clean_corr, 1.0)
        return np.clip(clean_corr, -1.0, 1.0)

    @classmethod
    def denoise_covariance(cls, sample_cov: np.ndarray, n_observations: int) -> np.ndarray:
        n_assets = sample_cov.shape[0]
        std_devs = np.sqrt(np.maximum(np.diag(sample_cov), 1e-8))
        inv_std = np.diag(1.0 / std_devs)
        corr = inv_std @ sample_cov @ inv_std

        denoised_corr = cls.denoise_correlation_rmt(corr, n_observations)
        denoised_cov = np.diag(std_devs) @ denoised_corr @ np.diag(std_devs)
        return (denoised_cov + denoised_cov.T) / 2.0

    @staticmethod
    def ledoit_wolf_shrinkage(returns: np.ndarray) -> Tuple[np.ndarray, float]:
        lw = LedoitWolf().fit(returns)
        shrunk_cov = lw.covariance_
        shrinkage_intensity = float(lw.shrinkage_)
        return shrunk_cov, shrinkage_intensity
