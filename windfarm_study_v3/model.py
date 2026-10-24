"""V3: fixed-capacity discrete siting, sourced benchmark cost, explicit envelope."""
import bootstrap
import json
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import qmc
from shapely.geometry import MultiPoint, box
from shapely.ops import unary_union
from evaluator import Scenario, wake_squared
from physical import physical_power

METRIC_COLUMNS = ['aep_GWh_per_year', 'benchmark_lcoe_USD2022_per_MWh',
                  'effective_envelope_km2', 'rooted_MST_m', 'gross_envelope_km2',
                  'excluded_envelope_km2', 'capex_USD2022', 'annual_cost_USD2022',
                  'electrical_capex_USD2022', 'reference_rooted_MST_m']


def candidate_pool(scene, m, seed=20261012):
    """Nested scrambled Sobol points; generator has no installed-count input."""
    if m < 2 or m >= 65536:
        raise ValueError('Candidate count must be between 2 and 65535')
    exponent = int(np.ceil(np.log2(max(32, 2 * m))))
    while exponent <= 20:
        raw = qmc.Sobol(2, scramble=True, seed=seed).random_base2(exponent)
        raw *= [scene.width, scene.height]
        allowed = scene.site_mask(raw, closed_obstacles=True)
        valid_indices = np.flatnonzero(allowed)
        if len(valid_indices) >= m:
            inspected = int(valid_indices[m - 1]) + 1
            return raw[allowed][:m], {'generator': 'scrambled_Sobol_2D', 'seed': seed,
                'installed_count_used': False, 'raw_points_inspected': inspected,
                'forbidden_points_filtered': inspected - m, 'retained_points': m,
                'anchor_injection': False}
        exponent += 1
    raise ValueError('Insufficient allowed candidate points')


def candidates(scene, m, n=None, seed=20261012):
    # n retained as a compatibility argument; it does not affect point placement.
    return candidate_pool(scene, m, seed)[0]


def independent_set_certificate(conflict):
    """A reproducible feasible lower bound, never an optimization fallback."""
    order = np.lexsort((np.arange(len(conflict)), conflict.sum(axis=1)))
    blocked = np.zeros(len(conflict), bool)
    selected = []
    for i in order:
        if not blocked[i]:
            selected.append(int(i))
            blocked |= conflict[i]
            blocked[i] = True
    return np.array(selected, int)


def repair_keys(keys, conflict, n, max_backtracks=20):
    keys = np.asarray(keys, float)
    if keys.shape != (len(conflict),) or not np.isfinite(keys).all():
        raise ValueError('Keys must be a finite M-vector')
    keys = np.clip(keys, 0, 1)
    order = np.argsort(keys, kind='stable')
    raw = np.sort(order[:n])
    raw_pairs = int(conflict[np.ix_(raw, raw)].sum() // 2)
    if raw_pairs == 0:
        return keys.copy(), raw, raw_pairs, True

    def fill(prefix, banned=None):
        chosen = list(prefix)
        blocked = conflict[chosen].any(axis=0) if chosen else np.zeros(len(keys), bool)
        blocked[chosen] = True
        if banned is not None:
            blocked[banned] = True
        for i in order:
            if not blocked[i]:
                chosen.append(int(i))
                blocked |= conflict[i]
                blocked[i] = True
                if len(chosen) == n:
                    break
        return chosen

    chosen = fill([])
    if len(chosen) < n:
        for i in range(min(max_backtracks, len(chosen))):
            base = chosen.copy()
            removed = base.pop(len(base) - 1 - i)
            trial = fill(base, removed)
            if len(trial) == n:
                chosen = trial
                break
    success = len(chosen) == n
    selected = np.sort(chosen) if success else raw
    if not success:
        return keys.copy(), selected, raw_pairs, False
    keep = np.zeros(len(keys), bool)
    keep[selected] = True
    new_order = np.r_[order[keep[order]], order[~keep[order]]]
    values = np.sort(keys)
    if np.any(np.diff(values) <= 1e-12):
        values = (np.arange(len(keys)) + .5) / len(keys)
    repaired = np.empty_like(keys)
    repaired[new_order] = values
    return repaired, selected, raw_pairs, True


def rooted_tree_length(xy, station):
    points = np.vstack([xy, station])
    # Coinciding terminal and substation are a single graph vertex.
    points = np.unique(points, axis=0)
    if len(points) < 2:
        return 0.
    return float(minimum_spanning_tree(cdist(points, points)).sum())


@lru_cache(maxsize=64)
def reference_layout(width, height, obstacles, spacing, n):
    """Geometric cost reference; never inserted into the candidate set."""
    step = 1.01 * spacing
    x, y = np.meshgrid(np.arange(step / 2, width, step), np.arange(step / 2, height, step))
    pool = np.c_[x.ravel(), y.ravel()]
    valid = np.ones(len(pool), bool)
    for a, b, c, d in obstacles:
        valid &= ~((pool[:, 0] >= a) & (pool[:, 0] <= c) & (pool[:, 1] >= b) & (pool[:, 1] <= d))
    pool = pool[valid]
    if len(pool) < n:
        raise ValueError('Not enough separated points for declared cost reference')
    seed = int(np.argmin(np.sum((pool - [width / 2, height / 2]) ** 2, axis=1)))
    ids = [seed]
    distance = np.sum((pool - pool[seed]) ** 2, axis=1)
    distance[seed] = -1
    while len(ids) < n:
        i = int(np.argmax(distance))
        ids.append(i)
        distance = np.minimum(distance, np.sum((pool - pool[i]) ** 2, axis=1))
        distance[ids] = -1
    return pool[ids]


class Economics:
    def __init__(self, params=None):
        self.params = dict(params) if params is not None else json.loads((bootstrap.ROOT / 'economics.json').read_text(encoding='utf-8'))
        p = self.params
        required = ['total_capex_per_kw', 'electrical_infrastructure_capex_per_kw',
                    'annual_opex_per_kw', 'real_fixed_charge_rate', 'nonwake_energy_multiplier']
        if not all(np.isfinite(p[k]) and p[k] > 0 for k in required):
            raise ValueError('Economic parameters must be finite and positive')
        if p['electrical_infrastructure_capex_per_kw'] >= p['total_capex_per_kw']:
            raise ValueError('Electrical budget must be part of total CAPEX')
        if p['nonwake_energy_multiplier'] > 1:
            raise ValueError('Energy multiplier cannot exceed one')

    def compute(self, rated_kw, n, length, reference_length, aep_gwh):
        if reference_length <= 0 or length < 0 or aep_gwh <= 0:
            raise ValueError('Invalid length or generation for cost evaluation')
        p = self.params
        capacity = rated_kw * n
        electrical = capacity * p['electrical_infrastructure_capex_per_kw'] * length / reference_length
        capex = capacity * (p['total_capex_per_kw'] - p['electrical_infrastructure_capex_per_kw']) + electrical
        annual = p['real_fixed_charge_rate'] * capex + capacity * p['annual_opex_per_kw']
        mwh = aep_gwh * 1000 * p['nonwake_energy_multiplier']
        return {'lcoe': annual / mwh, 'capex': capex, 'annual_cost': annual, 'electrical_capex': electrical}


class Farm:
    def __init__(self, scenario='obs_00', m=800, n=None, seed=20261012, economics=None):
        self.scene = Scenario.load(bootstrap.STANDARD / 'data' / 'Scenarios' / (scenario + '.xml'))
        self.m, self.n = int(m), self.scene.nominal_turbines if n is None else int(n)
        if not 2 <= self.n < self.m:
            raise ValueError('Require 2 <= installed N < candidate M')
        self.xy, self.candidate_metadata = candidate_pool(self.scene, self.m, seed)
        self.economics = Economics(economics)
        self.records, self.cache = [], {}
        self.prepare()
        self.certificate = independent_set_certificate(self.conflict)
        if len(self.certificate) < self.n:
            raise ValueError(f'No sufficient greedy feasibility certificate: {len(self.certificate)} < {self.n}; not a proof of infeasibility')

    def prepare(self):
        self.distance = cdist(self.xy, self.xy)
        self.conflict = self.distance < self.scene.min_spacing
        np.fill_diagonal(self.conflict, False)
        self.wake = None
        self.allowed = box(0, 0, self.scene.width, self.scene.height).difference(unary_union([box(*x) for x in self.scene.obstacles]))
        self.free_aep = float(physical_power(self.scene, np.zeros((24, self.n))).sum() * 8760 / 1e6)
        self.station = np.array([self.scene.width / 2, 0.])
        if not self.scene.site_mask(self.station[None], closed_obstacles=True)[0]:
            raise ValueError('Declared reference substation falls in a forbidden area')
        geometry = tuple(tuple(float(v) for v in row) for row in self.scene.obstacles)
        # Reference uses the XML geometry, independent of M and algorithm output.
        self.reference_xy = reference_layout(self.scene.width, self.scene.height, geometry, self.scene.min_spacing, self.n).copy()
        self.reference_length = rooted_tree_length(self.reference_xy, self.station)
        self.reference_cost = self.economics.compute(self.scene.rated_kw, self.n, self.reference_length, self.reference_length, self.free_aep)
        self.scales = np.array([self.free_aep, self.reference_cost['lcoe'], self.allowed.area / 1e6])

    def ensure_wake(self):
        if self.wake is None:
            self.wake = wake_squared(self.scene, self.xy)

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in ('distance', 'conflict', 'wake', 'allowed'):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.prepare()

    def case_metadata(self):
        return {'model_version': 'v3.0', 'scenario': self.scene.name, 'M': self.m, 'N': self.n,
                'xml_nominal_N': self.scene.nominal_turbines, 'installed_capacity_MW': self.n * self.scene.rated_kw / 1000,
                'site_km2': self.scene.width * self.scene.height / 1e6, 'allowed_km2': self.allowed.area / 1e6,
                'turbines_per_allowed_km2': self.n / (self.allowed.area / 1e6), 'selection_fraction': self.n / self.m,
                'spacing_m': self.scene.min_spacing, 'conflict_pairs': int(self.conflict.sum() // 2),
                'feasible_certificate_size': len(self.certificate), 'certificate_used_for_initialization_or_repair': False,
                'candidates': self.candidate_metadata, 'obstacles': self.scene.obstacles.tolist(),
                'station_m': self.station.tolist(), 'reference_rooted_MST_m': self.reference_length,
                'fixed_objective_scales': self.scales.tolist(), 'metric_columns': METRIC_COLUMNS,
                'elevation': None, 'area_definition': 'effective development envelope, zero buffer, not foundation footprint',
                'economic_parameters': self.economics.params}

    def evaluate(self, keys):
        raw = np.argsort(np.clip(keys, 0, 1), kind='stable')[:self.n]
        repaired, selected, raw_pairs, repaired_ok = repair_keys(keys, self.conflict, self.n)
        gaps = np.maximum(0, (self.scene.min_spacing - self.distance[np.ix_(selected, selected)]) / self.scene.min_spacing)
        cv = float(gaps[np.triu_indices(self.n, 1)].sum() / (self.n * (self.n - 1) / 2))
        packed = selected.astype('<u2').tobytes()
        hit = packed in self.cache
        if not hit:
            metrics = np.zeros(len(METRIC_COLUMNS))
            if cv == 0:
                self.ensure_wake()
                deficits = np.sqrt(self.wake[:, selected[:, None], selected[None, :]].sum(axis=2))
                energy = float(physical_power(self.scene, deficits).sum() * 8760 / 1e6)
                length = rooted_tree_length(self.xy[selected], self.station)
                cost = self.economics.compute(self.scene.rated_kw, self.n, length, self.reference_length, energy)
                hull = MultiPoint(self.xy[selected]).convex_hull
                gross = hull.area / 1e6
                area = hull.intersection(self.allowed).area / 1e6
                metrics = np.array([energy, cost['lcoe'], area, length, gross, gross - area,
                                    cost['capex'], cost['annual_cost'], cost['electrical_capex'], self.reference_length])
                if not np.isfinite(metrics).all() or energy <= 0:
                    raise ValueError('Invalid objective values')
            self.cache[packed] = (metrics, cv)
        metrics, cv = self.cache[packed]
        f = np.array([-metrics[0], metrics[1], metrics[2]]) / self.scales if cv == 0 else np.full(3, 1e6)
        replacements = self.n - len(np.intersect1d(raw, selected))
        self.records.append((packed, metrics.copy(), cv, raw_pairs, repaired_ok, hit, replacements))
        return repaired, f, cv

    def groups(self, rng, mode, size=100):
        if mode == 'random':
            order = rng.permutation(self.m)
            return [a for a in np.array_split(order, int(np.ceil(self.m / size)))]
        self.ensure_wake()
        weights = np.tensordot(self.scene.omega / self.scene.omega.sum(), self.wake, axes=(0, 0))
        weights += weights.T.copy()
        weights /= max(float(weights.max()), 1e-12)
        weights += self.conflict
        remaining = set(range(self.m))
        result = []
        while remaining:
            seed = int(rng.choice(sorted(remaining)))
            group = [seed]
            remaining.remove(seed)
            scores = weights[seed].copy()
            while remaining and len(group) < size:
                indices = np.array(sorted(remaining))
                chosen = int(indices[np.argmax(scores[indices])])
                group.append(chosen)
                remaining.remove(chosen)
                scores += weights[chosen]
            result.append(np.array(group))
        return result


def independent_check(scene, xy):
    from math import hypot, isfinite
    for i, (x, y) in enumerate(xy):
        if not (isfinite(x) and isfinite(y) and 0 <= x <= scene.width and 0 <= y <= scene.height):
            return False
        if any(a <= x <= c and b <= y <= d for a, b, c, d in scene.obstacles):
            return False
        if any(hypot(x - a, y - b) < scene.min_spacing - 1e-9 for a, b in xy[:i]):
            return False
    return True
