"""Exact Weibull integral of the declared nonnegative piecewise power curve.

The legacy Java-compatible scoring evaluator stays untouched. This module is
the study's physical objective, with normalized probabilities and cut-out.
"""
import bootstrap
import numpy as np
from scipy.special import gamma, gammaincc


def physical_power(scene, deficits):
    loss = np.clip(np.asarray(deficits, float), 0, 1)
    scale = scene.c[:, None] * (1 - loss)
    shape = scene.shape[:, None]
    safe = np.maximum(scale, 1e-30)
    slope, intercept = 140.86, -500.
    # The inherited linear ramp is slightly negative between 3.5 and
    # 500/140.86 m/s; physical generation must be nonnegative.
    low = max(scene.cut_in, -intercept / slope)
    with np.errstate(over='ignore', under='ignore'):
        t0 = (low / safe) ** shape
        t1 = (scene.rated_speed / safe) ** shape
        t2 = (scene.cut_out / safe) ** shape
    a = 1 + 1 / shape
    probability = np.exp(-t0) - np.exp(-t1)
    moment = safe * gamma(a) * (gammaincc(a, t0) - gammaincc(a, t1))
    expected = slope * moment + intercept * probability
    expected += scene.rated_kw * (np.exp(-t1) - np.exp(-t2))
    expected = np.where(scale > 0, np.maximum(expected, 0), 0)
    return expected * (scene.omega / scene.omega.sum())[:, None]
