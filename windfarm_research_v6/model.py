"""One overlap-aware objective definition shared by every V6 comparator."""
from collections import Counter, OrderedDict
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
from shapely.geometry import MultiPoint

from . import MODEL_VERSION
from .compat import ROOT, TerrainFarm, METRIC_COLUMNS, repair_keys, physical_power, tree_length
from .feasible import selected, token, encode, validate
from .lifecycle import FileLock
from .wake import wake_squared


class OverlapFarm(TerrainFarm):
    def __init__(self, scenario="obs_00", m=1600, n=400, seed=20261012,
                 terrain_kind="flat", initial_size=100, cache_root=None,
                 incremental=True, sum_cache_size=256):
        self.initial_size = initial_size
        self.initial_keys = []
        self.statistics = Counter()
        self.components = Counter()
        self.sum_cache = OrderedDict()
        self.sum_cache_size = sum_cache_size
        self.incremental = incremental
        self.kernel_cache_root = Path(cache_root) if cache_root else ROOT/"windfarm_research_v6/cache"
        self.kernel_metadata = None
        super().__init__(scenario=scenario, m=m, n=n, seed=seed, terrain_kind=terrain_kind)
        if not (self.scene.width == 7000 and self.scene.height == 14000):
            raise ValueError("The supplied synthetic DEM is restricted to a 7000x14000 m site")

    def ensure_wake(self):
        if self.wake is not None:
            return
        started = time.perf_counter()
        digest = hashlib.sha256()
        digest.update(Path(__file__).with_name("wake.py").read_bytes())
        digest.update(self.xy.astype("<f8").tobytes())
        digest.update(self.hub_z.astype("<f8").tobytes())
        digest.update(json.dumps([self.scene.radius, self.scene.decay, self.scene.thrust, MODEL_VERSION]).encode())
        key = digest.hexdigest()
        self.kernel_cache_root.mkdir(parents=True, exist_ok=True)
        path = self.kernel_cache_root/(key+".npy")
        hit = path.exists()
        with FileLock(path.with_suffix(".lock"), wait=True):
            if not path.exists():
                temp = path.with_name(path.name+f".{os.getpid()}.tmp")
                matrix = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float64,
                                                   shape=(24, self.m, self.m))
                wake_squared(self.scene, self.xy, self.hub_z, matrix)
                matrix.flush()
                del matrix
                os.replace(temp, path)
        self.wake = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.wake.shape != (24, self.m, self.m) or self.wake.dtype != np.float64:
            raise ValueError("Incompatible wake cache")
        self.kernel_metadata = {"key": key, "path": str(path), "read_only": True,
                                "cache_existed_before_lock": hit}
        self.components["kernel_setup_wall_seconds"] += time.perf_counter()-started

    def feasible(self, indices):
        return bool(validate(indices, self.conflict, self.n) and
                    self.scene.site_mask(self.xy[indices], closed_obstacles=True).all() and
                    self.terrain.allowed_mask(self.xy[indices]).all())

    def _put_sums(self, key, values):
        self.sum_cache[key] = values
        self.sum_cache.move_to_end(key)
        while len(self.sum_cache) > self.sum_cache_size:
            self.sum_cache.popitem(last=False)

    def _wake_sums(self, indices, parent=None, force_full=False):
        self.ensure_wake()
        key = token(indices)
        if not force_full and key in self.sum_cache:
            self.sum_cache.move_to_end(key)
            return self.sum_cache[key]
        use_delta = False
        if parent is not None and self.incremental and not force_full:
            parent = np.sort(np.asarray(parent, int))
            removed = np.setdiff1d(parent, indices)
            added = np.setdiff1d(indices, parent)
            parent_key = token(parent)
            use_delta = (parent_key in self.sum_cache and
                         len(removed) <= max(8, self.n//4))
        if use_delta:
            common = np.intersect1d(parent, indices)
            sums = np.empty((24, self.n), float)
            values = self.sum_cache[parent_key][:, np.searchsorted(parent, common)].copy()
            values -= self.wake[:, common[:, None], removed[None, :]].sum(axis=2)
            values += self.wake[:, common[:, None], added[None, :]].sum(axis=2)
            sums[:, np.searchsorted(indices, common)] = values
            sums[:, np.searchsorted(indices, added)] = self.wake[:, added[:, None], indices[None, :]].sum(axis=2)
            if np.any(sums < -1e-11):
                raise ArithmeticError("Negative squared wake sum beyond roundoff tolerance")
            sums = np.maximum(sums, 0.)
            self.statistics["incremental_wake_evaluations"] += 1
        else:
            sums = self.wake[:, indices[:, None], indices[None, :]].sum(axis=2)
            self.statistics["full_wake_evaluations"] += 1
        if not force_full:
            self._put_sums(key, sums)
        return sums

    def physical_metrics(self, indices, parent=None, force_full=False):
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu":
            raise ValueError("Candidate indices must be integers, without implicit rounding")
        indices = np.sort(indices.astype(int))
        if not self.feasible(indices):
            raise ValueError("Accurate evaluation requires a feasible layout of exactly N points")
        started = time.perf_counter()
        sums = self._wake_sums(indices, parent, force_full)
        energy = float(physical_power(self.scene, np.sqrt(sums)).sum()*8760/1e6)
        self.components["energy_wall_seconds"] += time.perf_counter()-started
        started = time.perf_counter()
        self.ensure_ground()
        length = tree_length(self.ground_dist, np.r_[indices, self.m], self.ground_points)
        cost = self.economics.compute(self.scene.rated_kw, self.n, length, self.reference_length, energy)
        self.components["cost_and_connection_wall_seconds"] += time.perf_counter()-started
        started = time.perf_counter()
        hull = MultiPoint(self.xy[indices]).convex_hull
        gross, area = hull.area/1e6, hull.intersection(self.allowed).area/1e6
        self.components["area_wall_seconds"] += time.perf_counter()-started
        values = np.array([energy, cost["lcoe"], area, length, gross, gross-area,
                           cost["capex"], cost["annual_cost"], cost["electrical_capex"], self.reference_length])
        if not np.isfinite(values).all() or energy <= 0:
            raise ValueError("Non-finite objective or nonpositive energy")
        return values

    def _record(self, original, keys, indices, raw_pairs, repaired_ok, parent=None):
        indices = np.sort(np.asarray(indices, int))
        gaps = np.maximum(0., (self.scene.min_spacing-self.distance[np.ix_(indices, indices)])/self.scene.min_spacing)
        cv = float(gaps[np.triu_indices(self.n, 1)].sum()/(self.n*(self.n-1)/2))
        key = token(indices)
        hit = key in self.cache
        if not hit:
            if cv == 0:
                metrics = self.physical_metrics(indices, parent)
                self.statistics["new_feasible_evaluations"] += 1
            else:
                metrics = np.zeros(len(METRIC_COLUMNS))
            self.cache[key] = metrics.copy(), cv
        metrics, cv = self.cache[key]
        f = np.array([-metrics[0], metrics[1], metrics[2]])/self.scales if cv == 0 else np.full(3, 1e6)
        replacements = self.n-len(np.intersect1d(selected(original, self.n), indices))
        self.records.append((key, metrics.copy(), cv, raw_pairs, repaired_ok, hit, replacements))
        self.diagnostics.append((float(np.mean(np.abs(keys-original)>1e-12)), float(np.linalg.norm(keys-original))))
        if len(self.initial_keys) < self.initial_size:
            self.initial_keys.append(keys.copy())
        self.statistics["requests"] += 1
        self.statistics["cache_hits"] += int(hit)
        self.statistics["feasible_requests"] += int(cv == 0)
        return keys.copy(), f, cv

    def evaluate(self, keys):
        """Frozen baseline decoder, new shared objective model."""
        original = np.clip(np.asarray(keys, float), 0., 1.)
        started = time.perf_counter()
        repaired, indices, raw_pairs, ok = repair_keys(original, self.conflict, self.n)
        self.components["decode_wall_seconds"] += time.perf_counter()-started
        return self._record(original, repaired, indices, raw_pairs, ok)

    def evaluate_layout(self, indices, scores, parent=None):
        """Problem-aware algorithms supply already-feasible physical decisions."""
        indices = np.asarray(indices)
        if indices.dtype.kind not in "iu":
            raise ValueError("Candidate indices must be integers, without implicit rounding")
        indices = np.sort(indices.astype(int))
        if not self.feasible(indices):
            raise ValueError("Problem-specific operator violated a hard constraint")
        original = np.clip(np.asarray(scores, float), 0., 1.)
        if original.shape != (self.m,) or not np.isfinite(original).all():
            raise ValueError("A finite M-dimensional operator vector is required")
        keys = encode(original, indices)
        raw = selected(original, self.n)
        raw_pairs = int(self.conflict[np.ix_(raw, raw)].sum()//2)
        return self._record(original, keys, indices, raw_pairs, True, parent)

    def case_metadata(self):
        value = super().case_metadata()
        value.update(model_version=MODEL_VERSION,
                     wake_scope="Area-weighted rotor overlap Jensen, per-source mean deficit then RSS; fixed CT; not pointwise rotor quadrature or terrain flow",
                     kernel=self.kernel_metadata, incremental_enabled=self.incremental,
                     timing_scope="Wall components can overlap (kernel setup inside energy); never sum them as disjoint pure CPU time")
        return value
