"""
Hierarchical Risk Parity (HRP - Marcos López de Prado 2016).
Machine-learning tree clustering approach that does not require matrix inversion.
Highly robust during crisis regimes when asset correlation matrices become ill-conditioned.
"""

from typing import List
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

class HierarchicalRiskParity:
    """
    Constructs an HRP portfolio through tree clustering, quasi-diagonalization,
    and recursive inverse-variance bisection.
    """

    @staticmethod
    def _correlation_distance(corr: np.ndarray) -> np.ndarray:
        """Computes distance matrix d_ij = sqrt(0.5 * (1 - rho_ij))."""
        d = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
        np.fill_diagonal(d, 0.0)
        return d

    @classmethod
    def _quasi_diagonalize(cls, link: np.ndarray) -> List[int]:
        """Sorts clustered assets by hierarchical distance."""
        link = link.astype(int)
        num_items = link[-1, 3]
        order = [link[-1, 0], link[-1, 1]]

        while any(idx >= num_items for idx in order):
            new_order = []
            for idx in order:
                if idx >= num_items:
                    row = idx - num_items
                    new_order.extend([link[row, 0], link[row, 1]])
                else:
                    new_order.append(idx)
            order = new_order
        return [int(x) for x in order]

    @classmethod
    def _get_cluster_var(cls, cov: np.ndarray, cluster_items: List[int]) -> float:
        """Calculates variance of a cluster under inverse-variance weighting."""
        sub_cov = cov[np.ix_(cluster_items, cluster_items)]
        diag = np.diag(sub_cov)
        inv_diag = 1.0 / np.maximum(diag, 1e-8)
        w = inv_diag / np.sum(inv_diag)
        return float(w.T @ sub_cov @ w)

    @classmethod
    def allocate(cls, covariance: np.ndarray) -> np.ndarray:
        """
        Computes HRP portfolio weights from asset covariance matrix.
        Returns weights vector summing to 1.0.
        """
        N = covariance.shape[0]
        if N == 1:
            return np.array([1.0])

        std = np.sqrt(np.diag(covariance))
        inv_std = 1.0 / np.maximum(std, 1e-8)
        corr = np.outer(inv_std, inv_std) * covariance
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        dist = cls._correlation_distance(corr)
        dist_condensed = squareform(dist, checks=False)
        link = sch.linkage(dist_condensed, method="single")

        sorted_indices = cls._quasi_diagonalize(link)

        weights = np.ones(N)
        clusters = [sorted_indices]

        while len(clusters) > 0:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    c1 = cluster[:mid]
                    c2 = cluster[mid:]

                    var1 = cls._get_cluster_var(covariance, c1)
                    var2 = cls._get_cluster_var(covariance, c2)

                    alpha = 1.0 - var1 / (var1 + var2)
                    weights[c1] *= alpha
                    weights[c2] *= (1.0 - alpha)

                    if len(c1) > 1:
                        new_clusters.append(c1)
                    if len(c2) > 1:
                        new_clusters.append(c2)
            clusters = new_clusters

        return weights / np.sum(weights)
