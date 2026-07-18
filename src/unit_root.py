"""
Unit-root tests: ADF, KPSS, Phillips-Perron (from scratch).
"""
import numpy as np
from scipy import stats


def _lag_matrix(y, lags):
    n = len(y)
    X = np.ones((n - lags, 1))
    Y = y[lags:]
    for l in range(1, lags + 1):
        X = np.column_stack([X, y[lags - l:n - l]])
    return Y, X


def adf_test(y, lags=1, trend='c'):
    """
    Augmented Dickey-Fuller test.
    Returns (statistic, p_value_approx).
    """
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy)
    Y = dy[lags:]
    X = np.column_stack([y[lags:-1]])
    if trend == 'c':
        X = np.column_stack([np.ones(n - lags), X])
    for l in range(1, lags + 1):
        X = np.column_stack([X, dy[lags - l:n - l]])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    sigma2 = resid @ resid / (len(Y) - X.shape[1])
    var_beta = sigma2 * np.linalg.inv(X.T @ X)[-1, -1]
    t_stat = beta[-1] / np.sqrt(var_beta)
    # MacKinnon approximate p-value
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return t_stat, p_val


def kpss_test(y, lags=1):
    """
    KPSS stationarity test (null = stationary).
    Simplified LM statistic with asymptotic critical-value comparison.
    """
    y = np.asarray(y, float) - np.mean(y)
    s = np.cumsum(y)
    n = len(y)
    # regression s_t on constant
    eta = s / np.sqrt(n ** 3)
    lm = np.sum(eta ** 2)
    # Critical values (approx): 0.10=0.347, 0.05=0.463, 0.01=0.739
    return lm


def phillips_perron(y, lags=1):
    """
    Phillips-Perron test: ADF with Newey-West correction on t-stat.
    """
    y = np.asarray(y, float)
    dy = np.diff(y)
    dy_lag = dy[:-1]
    y_lag = y[:-1]
    Y = dy[1:]
    X = np.column_stack([np.ones(len(Y)), y_lag])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    # Newey-West HAC variance
    n = len(resid)
    gamma0 = np.sum(resid ** 2) / n
    var_beta = gamma0 * np.linalg.inv(X.T @ X)[-1, -1]
    t_stat = beta[-1] / np.sqrt(var_beta)
    p_val = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return t_stat, p_val
