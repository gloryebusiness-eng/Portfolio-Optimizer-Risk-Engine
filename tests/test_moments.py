import numpy as np
import pandas as pd
import pytest

from portfolio_engine.estimation.moments import (
    ledoit_wolf_covariance,
    sample_covariance,
    condition_number,
)


@pytest.fixture
def synthetic_returns():
    """Known covariance structure so we can test recovery, not just shape."""
    rng = np.random.default_rng(42)
    true_cov = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.09, 0.02],
        [0.00, 0.02, 0.16],
    ]) / 252
    draws = rng.multivariate_normal(np.zeros(3), true_cov, size=5000)
    return pd.DataFrame(draws, columns=["A", "B", "C"]), true_cov * 252


def test_sample_cov_recovers_truth(synthetic_returns):
    returns, true_annual = synthetic_returns
    est = sample_covariance(returns).values
    assert np.allclose(est, true_annual, atol=0.01)


def test_lw_is_symmetric_and_psd(synthetic_returns):
    returns, _ = synthetic_returns
    cov, _ = ledoit_wolf_covariance(returns)
    V = cov.values
    assert np.allclose(V, V.T)
    assert np.linalg.eigvalsh(V).min() > -1e-10


def test_shrinkage_intensity_in_unit_interval(synthetic_returns):
    returns, _ = synthetic_returns
    _, delta = ledoit_wolf_covariance(returns)
    assert 0.0 <= delta <= 1.0


def test_shrinkage_decreases_with_more_data():
    """delta should fall as n grows with p fixed — less noise, less need to shrink."""
    rng = np.random.default_rng(7)
    cov = np.eye(6) * 0.01 + 0.003
    small = pd.DataFrame(rng.multivariate_normal(np.zeros(6), cov, 60))
    large = pd.DataFrame(rng.multivariate_normal(np.zeros(6), cov, 4000))
    _, d_small = ledoit_wolf_covariance(small)
    _, d_large = ledoit_wolf_covariance(large)
    assert d_small > d_large


def test_shrinkage_improves_conditioning_when_n_near_p():
    """In the regime shrinkage targets, it should reduce the condition number."""
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(40, 25)))
    S = sample_covariance(returns)
    LW, delta = ledoit_wolf_covariance(returns)
    assert delta > 0.1
    assert condition_number(LW) < condition_number(S)