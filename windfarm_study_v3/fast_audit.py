"""Independent vectorized geometry audit; no changes to optimization objectives."""
import bootstrap
import numpy as np
from scipy.spatial.distance import pdist
from model import independent_check as scalar_check


def independent_check(scene, xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('Expected n by 2 coordinates')
    if not np.isfinite(xy).all():
        return False
    x, y = xy[:, 0], xy[:, 1]
    if np.any((x < 0) | (x > scene.width) | (y < 0) | (y > scene.height)):
        return False
    for a, b, c, d in scene.obstacles:
        if np.any((a <= x) & (x <= c) & (b <= y) & (y <= d)):
            return False
    distances = pdist(xy, metric='euclidean')
    threshold = scene.min_spacing - 1e-9
    # Preserve the original math.hypot decision at floating point boundaries.
    # For ordinary layouts the compiled distance calculation is sufficient.
    if np.any(np.abs(distances - threshold) <= max(1., abs(threshold)) * 1e-12):
        return scalar_check(scene, xy)
    return not np.any(distances < threshold)
