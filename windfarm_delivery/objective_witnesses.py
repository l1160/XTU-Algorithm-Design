"""Fixed-layout cost controls and explicit empirical nonredundancy witnesses."""
from pathlib import Path
import csv
import json
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
import numpy as np
from algorithms import NDS
from model import independent_check
from evaluator import Scenario
from campaign_figures import table, check_complete


def run(src, out):
    src, out = Path(src), Path(out)
    check_complete(src)
    out.mkdir(parents=True, exist_ok=True)
    economics = json.loads((ROOT / "windfarm_study_v3/economics.json").read_text(encoding="utf-8"))
    with (ROOT / "windfarm_research_v4/results/objectives/information_content.csv").open(encoding="utf-8-sig") as stream:
        earlier = {r["name"]: r for r in csv.DictReader(stream)}
    coefficient = economics["real_fixed_charge_rate"]
    sensitivity, witnesses = [], []
    best = {}
    counter = 0
    for donepath in sorted(src.glob("*/DONE.json")):
        folder = donepath.parent
        if not (folder / "evaluations.npz").exists():
            continue
        done = json.loads(donepath.read_text(encoding="utf-8"))
        with np.load(folder / "evaluations.npz") as archive:
            selected = archive["selected"].copy()
            metrics = archive["metrics"].copy()
            cv = archive["cv"].copy()
            xy = archive["candidate_xy"].copy()
        ui = np.unique(selected, axis=0, return_index=True)[1]
        ui = ui[cv[ui] == 0]
        m = metrics[ui]
        capacity_kw = int(done["turbines"]) * 1500
        c0 = capacity_kw * (coefficient * (
            economics["total_capex_per_kw"] - economics["electrical_infrastructure_capex_per_kw"]
        ) + economics["annual_opex_per_kw"])
        electrical = coefficient * m[:, 8]
        assert np.allclose(m[:, 7], c0 + electrical, rtol=1e-13, atol=1e-6)
        assert np.allclose(m[:, 1], (c0 + electrical) / (1000 * m[:, 0]), rtol=1e-13, atol=1e-10)
        nd2 = set(map(int, NDS.do(np.column_stack([-m[:, 0], m[:, 2]]),
                                  only_non_dominated_front=True)))
        eps = 1e-5
        numeric = (np.log(c0 + np.exp(eps) * electrical) -
                   np.log(c0 + np.exp(-eps) * electrical)) / (2 * eps)
        assert np.allclose(numeric, electrical / (c0 + electrical), rtol=1e-7, atol=1e-9)
        fronts = {}
        for scale in (0., .8, 1., 1.2, 2.):
            lcoe = (c0 + scale * electrical) / (1000 * m[:, 0])
            nd = set(map(int, NDS.do(np.column_stack([-m[:, 0], lcoe, m[:, 2]]),
                                      only_non_dominated_front=True)))
            fronts[scale] = nd
            if scale == 0:
                assert nd == nd2
            added = nd - nd2
            sensitivity.append(dict(run=done["name"], M=done["candidates"], variant=done["variant"],
                seed=done["seed"], electrical_cost_multiplier=scale,
                unique_feasible_layouts=len(m), energy_area_front=len(nd2),
                three_objective_front=len(nd), cost_added_layouts=len(added),
                cost_added_fraction=len(added)/len(nd),
                mean_LCOE_relative_change_percent=float(np.mean(lcoe / m[:, 1] - 1) * 100)))
        assert len(fronts[1.] - nd2) == int(earlier[done["name"]]["nondominated_layouts_added_by_cost"])
        for scale in fronts:
            rr = next(r for r in reversed(sensitivity) if r["run"] == done["name"]
                      and r["electrical_cost_multiplier"] == scale)
            baseline_added = fronts[1.] - nd2
            rr["baseline_added_still_nondominated"] = len(baseline_added & fronts[scale])
            rr["baseline_added_count"] = len(baseline_added)
        candidates = []
        for a in sorted(fronts[1.] - nd2):
            dominates = ((m[:, 0] >= m[a, 0]) & (m[:, 2] <= m[a, 2]) &
                         ((m[:, 0] > m[a, 0]) | (m[:, 2] < m[a, 2])))
            ids = np.flatnonzero(dominates)
            assert len(ids)
            assert np.all(m[a, 1] < m[ids, 1])
            b = int(ids[np.argmax(m[ids, 1] / m[a, 1] - 1)])
            denominator = electrical[b] * m[a, 0] - electrical[a] * m[b, 0]
            assert denominator > 0
            threshold = c0 * (m[b, 0] - m[a, 0]) / denominator
            assert 0 <= threshold < 1
            cost_a = (c0 + threshold * electrical[a]) / m[a, 0]
            cost_b = (c0 + threshold * electrical[b]) / m[b, 0]
            assert np.isclose(cost_a, cost_b, rtol=1e-12)
            advantage = (m[b, 1] / m[a, 1] - 1) * 100
            candidates.append((advantage, a, b, threshold))
        if candidates:
            advantage, a, b, threshold = max(candidates)
            witness = dict(run=done["name"], M=done["candidates"], variant=done["variant"],
                seed=done["seed"], A_request_index_zero_based=int(ui[a]),
                B_request_index_zero_based=int(ui[b]), A_AEP_GWh=float(m[a, 0]),
                B_AEP_GWh=float(m[b, 0]), A_LCOE=float(m[a, 1]), B_LCOE=float(m[b, 1]),
                A_effective_area_km2=float(m[a, 2]), B_effective_area_km2=float(m[b, 2]),
                A_annual_cost=float(m[a, 7]), B_annual_cost=float(m[b, 7]),
                B_LCOE_premium_relative_to_A_percent=float(advantage),
                equal_LCOE_electrical_multiplier=float(threshold))
            witnesses.append(witness)
            size = int(done["candidates"])
            if size not in best or advantage > best[size]["advantage"]:
                best[size] = dict(advantage=advantage, witness=witness,
                    A=selected[ui[a]].copy(), B=selected[ui[b]].copy(), xy=xy, scenario=done["scenario"])
        counter += 1
        if counter % 12 == 0:
            print("OBJECTIVE CONTROLS", counter, "/72", flush=True)
    assert counter == 72 and len(witnesses) == 62
    for size, item in best.items():
        scene = Scenario.load(ROOT / "windflo_standard/data/Scenarios" / (item["scenario"] + ".xml"))
        for label in ("A", "B"):
            ids = item[label]
            assert independent_check(scene, item["xy"][ids])
            table(out / f"M{size}_{label}_coordinates.csv", [
                dict(candidate_index_zero_based=int(i), x_m=float(item["xy"][i, 0]),
                     y_m=float(item["xy"][i, 1])) for i in ids])
        (out / f"M{size}_witness.json").write_text(json.dumps(item["witness"], ensure_ascii=False, indent=2), encoding="utf-8")
    table(out / "fixed_layout_cost_sensitivity.csv", sensitivity)
    table(out / "nonredundancy_witnesses.csv", witnesses)
    aggregate = []
    for size in (800, 1600):
        for scale in (0., .8, 1., 1.2, 2.):
            rr = [r for r in sensitivity if r["M"] == size and r["electrical_cost_multiplier"] == scale]
            aggregate.append(dict(M=size, electrical_cost_multiplier=scale, runs=len(rr),
                runs_with_cost_added_layouts=sum(r["cost_added_layouts"] > 0 for r in rr),
                mean_cost_added_fraction=float(np.mean([r["cost_added_fraction"] for r in rr])),
                baseline_added_survivors=sum(r["baseline_added_still_nondominated"] for r in rr),
                baseline_added_count=sum(r["baseline_added_count"] for r in rr),
                mean_LCOE_relative_change_percent=float(np.mean([r["mean_LCOE_relative_change_percent"] for r in rr]))))
    table(out / "aggregate_cost_sensitivity.csv", aggregate)
    return dict(status="complete", runs=counter, scenarios_per_run=5, concrete_witnesses=len(witnesses),
        constant_cost_degeneracy_verified=True, baseline_objective_information_reproduced=True,
        annual_cost_and_LCOE_units_verified=True, numerical_elasticity_verified=True,
        pairwise_break_even_costs_verified=True, independent_geometries_checked=4,
        new_optimization_runs=0, scope="Fixed visited layouts only; no site-price calibration or reoptimization.")
