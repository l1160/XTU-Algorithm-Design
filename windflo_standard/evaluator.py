"""Python 3 port of d9w/WindFLO Java evaluator (MIT; see reference/LICENSE).

Legacy mode deliberately retains the benchmark's 15-degree multiplier and
above-rated infinite tail. Physical mode is an explicitly modified 2-D model.
Neither mode includes elevation, spatially varying wind, or turbine control.
"""
import bootstrap
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


@dataclass
class Scenario:
    name: str
    width: float
    height: float
    nominal_turbines: int
    wake_free_energy: float
    c: np.ndarray
    shape: np.ndarray
    omega: np.ndarray
    obstacles: np.ndarray
    radius: float = 38.5
    thrust: float = 0.8
    decay: float = 0.075
    rated_kw: float = 1500.0
    cut_in: float = 3.5
    rated_speed: float = 14.0
    cut_out: float = 20.0
    elevation: object = None

    @property
    def min_spacing(self):
        return 8 * self.radius

    @classmethod
    def load(cls, path):
        root = ET.parse(path).getroot()
        angles = root.findall('.//angle')
        if len(angles) != 24:
            raise ValueError('WindFLO reference requires 24 ordered 15-degree bins')
        arrays = [np.array([float(a.attrib[k]) for a in angles]) for k in ('c', 'k', 'omega')]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError('Non-finite wind resource')
        if np.any(arrays[0] <= 0) or np.any(arrays[1] <= 0) or np.any(arrays[2] < 0) or arrays[2].sum() <= 0:
            raise ValueError('Invalid Weibull parameters or frequencies')
        bounds = np.array([[float(o.attrib[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                           for o in root.findall('.//obstacle')]).reshape(-1, 4)
        value = lambda key: float(root.findtext('.//' + key))
        result = cls(Path(path).stem, value('Width'), value('Height'), int(value('NTurbines')),
                     value('WakeFreeEnergy'), *arrays, bounds)
        if result.width <= 0 or result.height <= 0 or result.wake_free_energy <= 0:
            raise ValueError('Invalid scenario dimensions or reference energy')
        return result

    def site_mask(self, xy, closed_obstacles=False):
        xy = np.asarray(xy, float)
        valid = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 1] >= 0)
        valid &= (xy[:, 0] <= self.width) & (xy[:, 1] <= self.height)
        for xmin, ymin, xmax, ymax in self.obstacles:
            if closed_obstacles:
                blocked = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
            else:
                blocked = (xy[:, 0] > xmin) & (xy[:, 0] < xmax) & (xy[:, 1] > ymin) & (xy[:, 1] < ymax)
            valid &= ~blocked
        return valid

    def feasible(self, xy, closed_obstacles=False):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not self.site_mask(xy, closed_obstacles).all():
            return False
        d = xy[:, None, :] - xy[None, :, :]
        squared = np.sum(d * d, axis=2)
        np.fill_diagonal(squared, np.inf)
        return bool(np.all(squared >= self.min_spacing ** 2))


def wake_squared(scenario, xy):
    """[direction, receiver, source] squared velocity deficits, as in Java."""
    xy = np.asarray(xy, float)
    dx = xy[:, 0, None] - xy[None, :, 0]
    dy = xy[:, 1, None] - xy[None, :, 1]
    result = np.empty((24, len(xy), len(xy)), float)
    rk = scenario.radius / scenario.decay
    for q in range(24):
        theta = np.deg2rad(q * 15 + 7.5)
        co, si = np.cos(theta), np.sin(theta)
        projection = dx * co + dy * si
        denominator = np.sqrt((dx + rk * co) ** 2 + (dy + rk * si) ** 2)
        with np.errstate(invalid='ignore', divide='ignore'):
            # Retain Java's acos behavior for out-of-domain roundoff (NaN -> false).
            beta = np.arccos((projection + rk) / denominator)
        deficit = (1 - np.sqrt(1 - scenario.thrust)) / (1 + scenario.decay / scenario.radius * np.abs(projection)) ** 2
        result[q] = np.where(beta < np.arctan(scenario.decay), deficit ** 2, 0)
        np.fill_diagonal(result[q], 0)
    return result


def cdf_java(speed, scale, shape):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = speed / scale
        exponent = np.power(ratio, shape)
        exponent = np.where(np.abs(shape - 2) < 0.0001, ratio * ratio, exponent)
        exponent = np.where(np.abs(shape - 1) < 0.0001, ratio, exponent)
        exponent = np.where(np.abs(shape) < 0.0001, 1.0, exponent)
        return 1 - np.exp(-exponent)


def power_by_direction(scenario, deficits, mode='legacy'):
    """Returns [direction, turbine]; legacy values are benchmark score units.

    physical: normalize sector probability, apply cut-out, clamp total deficit
    to [0,1]. Power remains the original piecewise curve and half-m/s quadrature.
    """
    if mode not in ('legacy', 'physical'):
        raise ValueError(mode)
    c = scenario.c[:, None] * (1 - (np.clip(deficits, 0, 1) if mode == 'physical' else deficits))
    shape = scenario.shape[:, None, None]
    edges = np.arange(3.5, 14.0 + 0.25, 0.5)
    cdf = cdf_java(edges[None, None, :], c[:, :, None], shape)
    midpoint_power = 140.86 * ((edges[1:] + edges[:-1]) / 2) - 500.0
    power = np.sum(np.diff(cdf, axis=2) * midpoint_power, axis=2)
    tail = 1 - cdf[:, :, -1]
    if mode == 'physical':
        tail -= 1 - cdf_java(scenario.cut_out, c, scenario.shape[:, None])
        weight = scenario.omega / scenario.omega.sum()
    else:
        weight = 15 * scenario.omega
    return (power + scenario.rated_kw * tail) * weight[:, None]


def competition_cost(energy_score, n):
    if energy_score <= 0:
        return np.finfo(float).max
    investment = (750000 * n + 8000000 * np.floor(n / 30)) * (0.666667 + 0.333333 * np.exp(-0.00174 * n * n))
    annual = (investment + 20000 * n) / ((1 - 1.03 ** -20) / 0.03)
    return annual / (8760 * energy_score) + 0.1 / n


class KusiakEvaluator:
    """Complete scoring API; no valid-layout return-0 placeholder."""
    def __init__(self, scenario):
        self.scenario = scenario

    def evaluate(self, xy):
        xy = np.asarray(xy, float)
        if not self.scenario.feasible(xy):
            return {'feasible': False, 'energy_score': 0.0, 'wake_free_ratio': 0.0,
                    'competition_cost': float(np.finfo(float).max), 'direction_outputs': None}
        deficits = np.sqrt(wake_squared(self.scenario, xy).sum(axis=2))
        energy = power_by_direction(self.scenario, deficits, 'legacy')
        total = float(energy.sum())
        return {'feasible': True, 'energy_score': total,
                'wake_free_ratio': total / (self.scenario.wake_free_energy * len(xy)),
                'competition_cost': float(competition_cost(total, len(xy))),
                'direction_outputs': energy}
