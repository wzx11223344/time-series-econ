import numpy as np
from src.unit_root import adf_test, kpss_test, phillips_perron

def test_adf_stationary():
    np.random.seed(0)
    y = np.cumsum(np.random.randn(200))  # random walk -> non-stationary
    stat, p = adf_test(y, lags=1)
    assert np.isfinite(stat)

def test_kpss():
    np.random.seed(1)
    y = np.random.randn(200)
    lm = kpss_test(y)
    assert lm >= 0
