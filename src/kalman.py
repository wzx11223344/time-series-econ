"""
Kalman filter (from scratch) — univariate and multivariate state-space.
"""
import numpy as np


def kalman_filter(y, F, H, Q, R, x0, P0):
    """
    Standard linear-Gaussian Kalman filter.

    F : state transition (m, m)
    H : observation (d, m)
    Q : process noise (m, m)
    R : observation noise (d, d)
    y : (T, d) observations
    """
    y = np.asarray(y, float)
    T = len(y)
    m = F.shape[0]
    states = np.zeros((T, m))
    for t in range(T):
        # Predict
        x_pred = F @ x0
        P_pred = F @ P0 @ F.T + Q
        # Update
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x0 = x_pred + K @ (y[t] - H @ x_pred)
        P0 = (np.eye(m) - K @ H) @ P_pred
        states[t] = x0
    return states
