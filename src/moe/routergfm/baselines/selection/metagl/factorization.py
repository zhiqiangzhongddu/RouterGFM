"""Latent factors of the performance matrix P (official MetaGL ``fit_graph_and_model_factors``).

Masked multiplicative NMF when P has missing entries, otherwise two
independent PCAs (rows and columns), exactly as the official code.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA

PCA_RANDOM_STATE = 100


def sparse_nmf(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    error_limit: float = 1e-6,
    fit_error_limit: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``X ~ A @ Y`` with ``A [n, k]``, ``Y [k, m]`` >= 0 fitted on observed entries (NaN = missing)."""
    eps = 1e-5
    X = np.array(X, dtype=np.float64)
    mask = ~np.isnan(X)
    X[~mask] = 0.0
    rng = rng if rng is not None else np.random.default_rng()
    A = np.maximum(rng.random((X.shape[0], int(k))), eps)
    Y = np.maximum(linalg.lstsq(A, X)[0], eps)
    masked_X = mask * X
    X_est_prev = A @ Y
    for i in range(1, int(max_iter) + 1):
        A = np.maximum(A * (masked_X @ Y.T) / ((mask * (A @ Y)) @ Y.T + eps), eps)
        Y = np.maximum(Y * (A.T @ masked_X) / (A.T @ (mask * (A @ Y)) + eps), eps)
        if i % 5 == 0 or i == 1 or i == max_iter:
            X_est = A @ Y
            fit_residual = np.sqrt(np.sum((mask * (X_est_prev - X_est)) ** 2))
            X_est_prev = X_est
            if linalg.norm(mask * (X - X_est), ord="fro") < error_limit or fit_residual < fit_error_limit:
                break
    return A, Y


def factorize(P: np.ndarray, k: int, rng: np.random.Generator, nmf_max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, str]:
    """``(U [n, k], V [m, k], kind)``: ``sparse_nmf`` if P has NaN, else PCA of P and of P^T."""
    if np.isnan(P).any():
        A, Y = sparse_nmf(P, k, max_iter=nmf_max_iter, rng=rng)
        return A, Y.T, "sparse_nmf"
    U = PCA(n_components=int(k), random_state=PCA_RANDOM_STATE).fit_transform(P)
    V = PCA(n_components=int(k), random_state=PCA_RANDOM_STATE).fit_transform(P.T)
    return U, V, "pca"


__all__ = ["factorize", "sparse_nmf"]
