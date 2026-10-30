"""Small numeric helpers."""
import numpy as np


def interp_nan(x, max_gap=None):
    """Linear fill of NaN runs; runs longer than max_gap frames stay NaN."""
    x = np.asarray(x, float).copy()
    n = np.isnan(x)
    if n.all() or not n.any():
        return x
    filled = np.interp(np.flatnonzero(n), np.flatnonzero(~n), x[~n])
    if max_gap is None:
        x[n] = filled
        return x
    idx = np.flatnonzero(n)
    full = x.copy()
    full[idx] = filled
    t = 0
    while t < len(x):
        if n[t]:
            u = t
            while u < len(x) and n[u]:
                u += 1
            inside = t > 0 and u < len(x)              # do not extrapolate before the first / after the last box
            if inside and u - t <= max_gap:
                x[t:u] = full[t:u]
            t = u
        else:
            t += 1
    return x
