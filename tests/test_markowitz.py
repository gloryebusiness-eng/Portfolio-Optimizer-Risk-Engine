import numpy as np
import pandas as pd
import pytest

from portfolio_engine.optimize.markowitz import (
    efficient_frontier,
    max_sharpe_portfolio,
    min_variance_portfolio,
    portfolio_sharpe,
    portfolio_volatility,
)


@pytest.fixture
def toy_moments():
    assets = ["LOW", "MID", "HIGH"]
    mu = pd.Series([0.04, 0.08, 0.12], index=assets)
    cov = pd.DataFrame(
        np.array([
            [0.010, 0.000, 0.000],
            [0.000, 0.040, 0.000],
            [0.000, 0.000, 0.090],
        ]),
        index=assets, columns=assets,
    )
    return mu, cov


def test_weights_sum_to_one(toy_moments):
    mu, cov = toy_moments
    for w in (min_variance_portfolio(cov), max_sharpe_portfolio(mu, cov)):
        assert np.isclose(w.sum(), 1.0)


def test_long_only_bounds_respected(toy_moments):
    mu, cov = toy_moments
    w = max_sharpe_portfolio(mu, cov)
    assert (w >= -1e-9).all() and (w <= 1 + 1e-9).all()


def test_min_variance_beats_equal_weight(toy_moments):
    """Definitional: no portfolio can have lower variance than the minimum."""
    _, cov = toy_moments
    w_mv = min_variance_portfolio(cov).values
    w_eq = np.repeat(1 / 3, 3)
    assert portfolio_volatility(w_mv, cov.values) <= portfolio_volatility(w_eq, cov.values) + 1e-12


def test_min_variance_tilts_toward_low_vol_asset(toy_moments):
    """With uncorrelated assets, weights should scale as inverse variance."""
    _, cov = toy_moments
    w = min_variance_portfolio(cov)
    assert w["LOW"] > w["MID"] > w["HIGH"]
    expected = (1 / np.diag(cov.values)) / (1 / np.diag(cov.values)).sum()
    assert np.allclose(w.values, expected, atol=1e-4)


def test_max_sharpe_is_maximal(toy_moments):
    """Beat 500 random feasible portfolios."""
    mu, cov = toy_moments
    best = portfolio_sharpe(max_sharpe_portfolio(mu, cov).values, mu.values, cov.values)
    rng = np.random.default_rng(3)
    for _ in range(500):
        w = rng.dirichlet(np.ones(3))
        assert portfolio_sharpe(w, mu.values, cov.values) <= best + 1e-6


def test_frontier_is_monotone_in_risk(toy_moments):
    """Higher target return must require weakly higher volatility."""
    mu, cov = toy_moments
    f = efficient_frontier(mu, cov, n_points=25)
    assert len(f) > 10
    assert (f["volatility"].diff().dropna() > -1e-8).all()