"""Posthoc robustness checks on all completed confirmation runs; no search."""
from pathlib import Path
import hashlib
import itertools
import json
import statistics
from campaign_figures import check_complete, csvrows, table
import numpy as np
from pymoo.indicators.hv import HV


def area_union(points, reference=1.1):
    area, best = 0., reference
    for x, y in sorted(map(tuple, points)):
        if x < reference and y < best:
            area += (reference - x) * (best - y)
            best = y
    return float(area)


def run(src, out):
    src, out = Path(src), Path(out)
    cfg, original = check_complete(src)
    assert len(original) == 320
    bounds = json.loads((src / "analysis/normalization.json").read_text(encoding="utf-8"))
    sensitivity, secondary, contrasts = [], [], []
    for n, row in enumerate(original, 1):
        with np.load(src / row["name"] / "evaluations.npz") as saved:
            f = saved["metrics"][:, :3] * [-1, 1, 1]
            valid = saved["cv"] == 0
            indices = saved["pareto_request_indices"]
        assert len(f) == 10000 and valid[indices].all()
        b = bounds[row["case"]]
        lo, hi = np.array(b["ideal"]), np.array(b["nadir"])
        width = np.where(hi > lo, hi - lo, 1)
        normalized = (f - lo) / width
        front = normalized[indices]
        values = {}
        for ref in (1.05, 1.1, 1.2):
            value = float(HV(ref_point=np.full(3, ref))(front))
            if ref == 1.1:
                assert abs(value - float(row["HV"])) < 1e-10
            sensitivity.append(dict(case=row["case"], variant=row["variant"], seed=row["seed"],
                                    reference=ref, HV=value))
        for label, dims in [("energy_area", [0, 2]), ("cost_area", [1, 2])]:
            value = float(HV(ref_point=np.full(2, 1.1))(front[:, dims]))
            # Independently integrate all visited feasible points, not just the
            # stored 3-D front; this also checks that projection is sufficient.
            independent = area_union(normalized[valid][:, dims])
            assert abs(value - independent) < 1e-10
            values["HV2_" + label] = value
        energy_area = f[indices][:, [0, 2]]
        retained = np.ones(len(indices), bool)
        for i, point in enumerate(energy_area):
            retained[i] = not np.any(np.all(energy_area <= point, axis=1) &
                                    np.any(energy_area < point, axis=1))
        extra = int((~retained).sum())
        secondary.append(dict(case=row["case"], variant=row["variant"], seed=row["seed"],
                              HV3_primary=float(row["HV"]), **values,
                              three_objective_nondominated=len(indices),
                              additional_layouts_retained_by_cost=extra,
                              additional_fraction=extra / len(indices)))
        if n % 40 == 0:
            print("METRIC REVIEW", n, "/320", flush=True)
    aggregate = []
    for case in bounds:
        for variant in cfg["variants"]:
            rr = [r for r in secondary if r["case"] == case and r["variant"] == variant]
            assert len(rr) == 10
            item = dict(case=case, variant=variant, n=10)
            for field in ("HV3_primary", "HV2_energy_area", "HV2_cost_area", "additional_fraction"):
                item[field + "_mean"] = statistics.mean(r[field] for r in rr)
                item[field + "_sd"] = statistics.stdev(r[field] for r in rr)
            item["runs_with_additional_cost_layouts"] = sum(r["additional_layouts_retained_by_cost"] > 0 for r in rr)
            for ref in (1.05, 1.1, 1.2):
                item[f"HV3_ref{ref}_mean"] = statistics.mean(
                    r["HV"] for r in sensitivity if r["case"] == case and
                    r["variant"] == variant and r["reference"] == ref)
            aggregate.append(item)
    # Independent exhaustive implementation of the existing 20 primary tests;
    # no new inferential family and no altered primary significance threshold.
    previous = csvrows(src / "analysis/paired_comparisons.csv")
    for row in previous:
        paired = [r for r in original if r["case"] == row["case"]]
        by = {(r["variant"], int(r["seed"])): float(r["HV"]) for r in paired}
        diff = [by[row["treatment"], s] - by[row["control"], s] for s in cfg["seeds"]]
        observed = abs(statistics.mean(diff))
        exceed = sum(abs(sum(sign*d for sign, d in zip(signs, diff))/len(diff)) >= observed-1e-14
                     for signs in itertools.product((-1, 1), repeat=len(diff)))
        p = exceed / 2**len(diff)
        assert abs(p-float(row["paired_signflip_p"])) < 1e-12
        contrasts.append(dict(row, independently_recomputed_p=p))
    order = sorted(range(len(contrasts)), key=lambda i: contrasts[i]["independently_recomputed_p"])
    adjusted = 0.
    for rank, i in enumerate(order):
        adjusted = max(adjusted, min(1., (len(order)-rank)*contrasts[i]["independently_recomputed_p"]))
        assert abs(adjusted-float(contrasts[i]["Holm_adjusted_p"])) < 1e-12
        contrasts[i]["independently_recomputed_Holm_p"] = adjusted
    table(out / "reference_sensitivity.csv", sensitivity)
    table(out / "secondary_per_run.csv", secondary)
    table(out / "secondary_aggregate.csv", aggregate)
    table(out / "verified_primary_tests.csv", contrasts)
    (out / "source_hashes.json").write_text(json.dumps({
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__), src / "analysis/per_run_metrics.csv",
            src / "analysis/paired_comparisons.csv", src / "analysis/normalization.json"]},
        indent=2), encoding="utf-8")
    return dict(status="complete", runs=320, primary_HV_rechecks=320,
                independent_all_points_2D_HV_checks=640, reference_points=[1.05, 1.1, 1.2],
                primary_tests_independently_verified=20, new_significance_tests=0,
                new_optimization_runs=0,
                scope="Posthoc local metric robustness and empirical objective nonredundancy, not reoptimization or real-site model calibration")
