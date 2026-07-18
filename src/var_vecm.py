"""
VAR / VECM estimation with Impulse Response (IRF) and Forecast Error
Variance Decomposition (FEVD) — from scratch.
"""
import numpy as np
from scipy.linalg import cholesky


def estimate_var(Y, p=1):
    """
    Y : (T, k) endogenous variables
    Returns coefficient matrices (k, k, p) and residuals.
    """
    Y = np.asarray(Y, float)
    T, k = Y.shape
    # Build lagged design
    Z = np.column_stack([Y[p - l - 1:T - l - 1] for l in range(p)])
    Yt = Y[p:]
    B = np.linalg.lstsq(Z, Yt, rcond=None)[0]  # (p*k, k)
    Bmat = np.stack([B[i * k:(i + 1) * k] for i in range(p)], axis=0)  # (p, k, k)
    resid = Yt - Z @ B
    return Bmat, resid


def irf(Bmat, steps=10, ident='cholesky'):
    """
    Orthogonalized impulse response functions.
    Returns (steps+1, k, k) array.
    """
    p, k, _ = Bmat.shape
    # companion matrix
    comp = np.zeros((p * k, p * k))
    comp[:k] = Bmat.reshape(k, p * k)
    for i in range(1, p):
        comp[i * k:(i + 1) * k, (i - 1) * k:i * k] = np.eye(k)
    # shock impact
    cov = np.eye(k)
    if ident == 'cholesky':
        try:
            L = cholesky(cov, lower=True)
        except np.linalg.LinAlgError:
            L = np.eye(k)
    else:
        L = np.eye(k)
    irfs = np.zeros((steps + 1, k, k))
    irfs[0] = L
    for s in range(1, steps + 1):
        power = np.linalg.matrix_power(comp, s)
        irfs[s] = power[:k] @ L
    return irfs


def fevd(Bmat, steps=10):
    """
    Forecast Error Variance Decomposition.
    """
    irfs = irf(Bmat, steps)
    fevd = np.zeros((steps + 1, k, k))
    for s in range(steps + 1):
        for shock in range(k):
            contrib = irfs[s, :, shock] ** 2
            total = np.sum(irfs[s] ** 2, axis=1)
            fevd[s, :, shock] = contrib / total
    return fevd
