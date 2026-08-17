"""Mean-variance portfolio optimization.

The optimizer is deliberately thin. It accepts whatever moments the
estimation layer hands it and does not know or care how they were
produced. All the fragility in mean-variance optimization lives in the
inputs, not here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def portfolio_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                     rf: float = 0.0) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return 0.0
    return (portfolio_return(weights, mu) - rf) / vol


def _base_constraints() -> list[dict]:
    """Fully invested: weights sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _solve(objective, n_assets: int, constraints: list[dict],
           bounds: tuple[float, float]) -> np.ndarray:
    x0 = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[bounds] * n_assets,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Optimizer failed to converge: {result.message}")
    return result.x


def min_variance_portfolio(cov: pd.DataFrame,
                           bounds: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Global minimum-variance portfolio.

    Depends only on the covariance matrix — no expected returns required.
    This is why it is far more stable out-of-sample than the tangency
    portfolio: it never touches the noisiest input.
    """
    C = cov.values
    w = _solve(lambda w: w @ C @ w, len(C), _base_constraints(), bounds)
    return pd.Series(w, index=cov.index, name="min_variance")


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0,
                         bounds: tuple[float, float] = (0.0, 1.0)) -> pd.Series:
    """Tangency portfolio — maximum Sharpe ratio.

    Minimizes the negative Sharpe ratio directly. Note this is a
    non-convex objective; SLSQP finds a local optimum. With long-only
    bounds and a well-conditioned covariance matrix this is reliable in
    practice, but it is a real caveat, not a formality.
    """
    m, C = mu.values, cov.values
    w = _solve(
        lambda w: -portfolio_sharpe(w, m, C, rf),
        len(m),
        _base_constraints(),
        bounds,
    )
    return pd.Series(w, index=mu.index, name="max_sharpe")


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 50,
                       bounds: tuple[float, float] = (0.0, 1.0)) -> pd.DataFrame:
    """Trace minimum-variance portfolios across a range of target returns."""
    m, C = mu.values, cov.values
    lo = min_variance_portfolio(cov).values @ m
    hi = m.max()

    rows = []
    for target in np.linspace(lo, hi, n_points):
        cons = _base_constraints() + [
            {"type": "eq", "fun": lambda w, t=target: w @ m - t}
        ]
        try:
            w = _solve(lambda w: w @ C @ w, len(m), cons, bounds)
        except RuntimeError:
            continue
        rows.append({
            "target_return": target,
            "volatility": portfolio_volatility(w, C),
            "sharpe": portfolio_sharpe(w, m, C),
            **dict(zip(mu.index, w)),
        })
    return pd.DataFrame(rows)