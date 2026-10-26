"""Analytic rotor-overlap Jensen kernel with declared area-averaged deficits.

Area-weighting each source and then RSS-combining is an engineering proxy.
It is not pointwise rotor quadrature, a terrain flow solver, or FLORIS parity.
"""
import numpy as np


def overlap_fraction(rotor_radius, wake_radius, separation):
    r, w, d = np.broadcast_arrays(np.asarray(rotor_radius, float),
                                   np.asarray(wake_radius, float),
                                   np.asarray(separation, float))
    if not (np.isfinite(r).all() and np.isfinite(w).all() and np.isfinite(d).all()):
        raise ValueError("Non-finite circle geometry")
    if np.any(r <= 0) or np.any(w <= 0) or np.any(d < 0):
        raise ValueError("Positive radii and nonnegative separation required")
    result = np.zeros(r.shape, float)
    contained = d <= np.abs(w-r)
    result[contained] = (np.minimum(r[contained], w[contained])/r[contained])**2
    partial = (d < r+w) & ~contained
    a, b, c = r[partial], w[partial], d[partial]
    u = np.clip((c*c+a*a-b*b)/(2*c*a), -1., 1.)
    v = np.clip((c*c+b*b-a*a)/(2*c*b), -1., 1.)
    product = np.maximum((-c+a+b)*(c+a-b)*(c-a+b)*(c+a+b), 0.)
    area = a*a*np.arccos(u)+b*b*np.arccos(v)-.5*np.sqrt(product)
    result[partial] = area/(np.pi*a*a)
    return np.clip(result, 0., 1.)


def wake_squared(scene, xy, hub_z=None, out=None):
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite x/y coordinates")
    n = len(xy)
    height = np.zeros(n) if hub_z is None else np.asarray(hub_z, float)
    if height.shape != (n,) or not np.isfinite(height).all():
        raise ValueError("Invalid hub heights")
    result = np.empty((24, n, n), float) if out is None else out
    if result.shape != (24, n, n) or result.dtype != np.dtype("float64"):
        raise ValueError("Kernel buffer must be float64 with shape (24,M,M)")
    dx = xy[:, 0, None]-xy[None, :, 0]
    dy = xy[:, 1, None]-xy[None, :, 1]
    dz = height[:, None]-height[None, :]
    for q in range(24):
        angle = np.deg2rad(q*15+7.5)
        along = dx*np.cos(angle)+dy*np.sin(angle)
        lateral = -dx*np.sin(angle)+dy*np.cos(angle)
        distance = np.maximum(along, 0.)
        radius = scene.radius+scene.decay*distance
        alpha = overlap_fraction(scene.radius, radius, np.hypot(lateral, dz))
        delta = (1-np.sqrt(1-scene.thrust))/(1+scene.decay*distance/scene.radius)**2
        result[q] = np.where(along > 0, (alpha*delta)**2, 0.)
        np.fill_diagonal(result[q], 0.)
    return result
