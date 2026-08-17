"""Expected return and covariance estimators.

Every estimator here has the same signature — returns DataFrame in,
annualized array out — so the optimizer never knows which one it got.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def mean_historical_return(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    """Annualized arithmetic mean of historical returns.

    This is the naive estimator. It is here as a baseline and as a
    cautionary tale: sample means are extremely noisy, and the optimizer
    will happily overfit to them.
    """
    return returns.mean() * periods


def sample_covariance(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    return returns.cov() * periods


def ledoit_wolf_covariance(
    returns: pd.DataFrame, periods: int = TRADING_DAYS
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    Returns (shrunk_covariance, shrinkage_intensity).

    The sample covariance matrix is unbiased but high-variance: with N
    assets you estimate N(N+1)/2 parameters from limited data, and the
    extreme eigenvalues are systematically biased (largest too large,
    smallest too small). Shrinking toward a structured target trades a
    little bias for a large variance reduction.
    """
    X = returns.values
    n, p = X.shape
    if n <= 1:
        raise ValueError("Need more than one observation.")

    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n  # MLE covariance, per Ledoit-Wolf convention

    # Target: constant-correlation matrix with sample variances preserved.
    var = np.diag(S)
    std = np.sqrt(var)
    outer_std = np.outer(std, std)
    corr = S / outer_std
    off_diag = corr[~np.eye(p, dtype=bool)]
    r_bar = off_diag.mean()

    F = r_bar * outer_std
    np.fill_diagonal(F, var)

    # pi: sum of asymptotic variances of the sample covariance entries.
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / n - S ** 2
    pi_hat = pi_mat.sum()

    # rho: covariance between the estimation errors of S and F.
    term = ((Xc ** 3).T @ Xc) / n - var[:, None] * S
    rho_diag = np.diag(pi_mat).sum()
    theta = term - var[:, None] * S * 0  # keep shapes explicit
    ratios = np.outer(1.0 / std, std)
    rho_off = (r_bar / 2.0) * (ratios * term + ratios.T * term.T)
    np.fill_diagonal(rho_off, 0.0)
    rho_hat = rho_diag + rho_off.sum()

    # gamma: squared Frobenius distance between sample and target.
    gamma_hat = np.linalg.norm(S - F, "fro") ** 2

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(np.clip(kappa / n, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    return pd.DataFrame(shrunk * periods, index=returns.columns, columns=returns.columns), delta


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue.

    High condition number means the matrix is near-singular: small changes
    in the inputs produce large changes in the inverse, and the optimizer
    depends on that inverse.
    """
    eigs = np.linalg.eigvalsh(cov.values)
    return float(eigs.max() / eigs.min())