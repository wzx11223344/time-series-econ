"""
Bai-Perron structural break detection (from scratch) — global minimum of
sum of squared residuals across partition points.
"""
import numpy as np
import itertools


def bai_perron(y, X, max_breaks=3, min_seg=2):
    """
    y : (T,) dependent variable
    X : (T, k) regressors (common across segments)
    Returns best partition (list of break indices).
    """
    y = np.asarray(y, float); X = np.asarray(X, float)
    T = len(y)
    best = None
    best_ssr = np.inf
    for nbreaks in range(1, max_breaks + 1):
        # enumerate feasible break positions
        positions = range(min_seg, T - min_seg * nbreaks + 1)
        candidates = [c for c in positions]
        if len(candidates) < nbreaks:
            continue
        for combo in itertools.combinations(candidates, nbreaks):
            ssr = 0.0
            prev = 0
            ok = True
            for br in list(combo) + [T]:
                seg_y = y[prev:br]
                seg_X = X[prev:br]
                if len(seg_y) < min_seg:
                    ok = False; break
                beta, *_ = np.linalg.lstsq(seg_X, seg_y, rcond=None)
                resid = seg_y - seg_X @ beta
                ssr += resid @ resid
                prev = br
            if ok and ssr < best_ssr:
                best_ssr = ssr
                best = combo
    return best, best_ssr
