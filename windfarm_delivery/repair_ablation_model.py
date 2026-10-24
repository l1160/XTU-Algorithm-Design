"""Isolated repair-policy adapter; frozen V3/V4 code is never edited."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
from shapely.geometry import MultiPoint
from variants import DiagnosticFarm
from model import rooted_tree_length, METRIC_COLUMNS
from physical import physical_power


def decode(keys, conflict, n, mode):
    """Mirror frozen repair_keys with explicit operation counters."""
    if mode not in ("none", "greedy", "bounded_backtrack"):
        raise ValueError("Unknown repair treatment")
    keys = np.asarray(keys, float)
    if keys.shape != (len(conflict),) or not np.isfinite(keys).all():
        raise ValueError("Keys must be a finite M-vector")
    keys = np.clip(keys, 0, 1)
    order = np.argsort(keys, kind="stable")
    raw = np.sort(order[:n])
    raw_pairs = int(conflict[np.ix_(raw, raw)].sum() // 2)
    details = dict(mode=mode, raw_conflict_pairs=raw_pairs, greedy_invoked=False,
                   greedy_fill_size=None, backtrack_attempts=0, backtrack_success=False)
    if raw_pairs == 0 or mode == "none":
        return keys.copy(), raw, raw_pairs, raw_pairs == 0, details

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
    details.update(greedy_invoked=True, greedy_fill_size=len(chosen))
    if len(chosen) < n and mode == "bounded_backtrack":
        for i in range(min(20, len(chosen))):
            base = chosen.copy()
            removed = base.pop(len(base) - 1 - i)
            trial = fill(base, removed)
            details["backtrack_attempts"] += 1
            if len(trial) == n:
                chosen = trial
                details["backtrack_success"] = True
                break
    success = len(chosen) == n
    selected = np.sort(chosen) if success else raw
    if not success:
        return keys.copy(), selected, raw_pairs, False, details
    keep = np.zeros(len(keys), bool)
    keep[selected] = True
    new_order = np.r_[order[keep[order]], order[~keep[order]]]
    values = np.sort(keys)
    if np.any(np.diff(values) <= 1e-12):
        values = (np.arange(len(keys)) + .5) / len(keys)
    repaired = np.empty_like(keys)
    repaired[new_order] = values
    return repaired, selected, raw_pairs, True, details


class RepairFarm(DiagnosticFarm):
    def __init__(self, *args, repair_mode, shared_initial_requests=100, **kwargs):
        self.repair_mode = repair_mode
        self.shared_initial_requests = shared_initial_requests
        self.repair_details = []
        super().__init__(*args, **kwargs)

    def evaluate(self, keys):
        started = time.perf_counter()
        original = np.clip(np.asarray(keys, float), 0, 1)
        warm = len(self.records) < self.shared_initial_requests
        mode = "bounded_backtrack" if warm else self.repair_mode
        repaired, selected, raw_pairs, ok, details = decode(keys, self.conflict, self.n, mode)
        decode_seconds = time.perf_counter() - started
        gaps = np.maximum(0, (self.scene.min_spacing - self.distance[np.ix_(selected, selected)]) / self.scene.min_spacing)
        cv = float(gaps[np.triu_indices(self.n, 1)].sum() / (self.n * (self.n - 1) / 2))
        packed = selected.astype("<u2").tobytes()
        hit = packed in self.cache
        physical_evaluated = not hit and cv == 0
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
                metrics = np.array([energy, cost["lcoe"], area, length, gross, gross-area,
                    cost["capex"], cost["annual_cost"], cost["electrical_capex"], self.reference_length])
                if not np.isfinite(metrics).all() or energy <= 0:
                    raise ValueError("Invalid objective values")
            self.cache[packed] = (metrics, cv)
        metrics, cv = self.cache[packed]
        f = np.array([-metrics[0], metrics[1], metrics[2]]) / self.scales if cv == 0 else np.full(3, 1e6)
        raw = np.argsort(original, kind="stable")[:self.n]
        replacements = self.n - len(np.intersect1d(raw, selected))
        self.records.append((packed, metrics.copy(), cv, raw_pairs, ok, hit, replacements))
        self.diagnostics.append((float(np.mean(np.abs(repaired-original)>1e-12)), float(np.linalg.norm(repaired-original))))
        self.repair_details.append(dict(details, shared_initialization=warm,
            decode_seconds=decode_seconds, evaluation_seconds=time.perf_counter()-started,
            physical_evaluated=physical_evaluated))
        return repaired, f, cv
