"""Equal-unique-layout secondary truncation of complete confirmatory trajectories."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"windfarm_research_v4"))
import bootstrap_v4
import numpy as np
from scipy.stats import rankdata
from pymoo.indicators.hv import HV
from algorithms import NDS
from campaign_figures import check_complete, csvrows, table


def run(src,out):
    src,out=Path(src),Path(out)
    cfg,original=check_complete(src)
    assert cfg["stage"]=="independent_seed_mechanism_confirmation"
    assert len(original)==320
    bounds=json.loads((src/"analysis/normalization.json").read_text(encoding="utf-8"))
    rows,aggregate,thresholds=[],[],[]
    for case in sorted(bounds):
        selected=[r for r in original if r["case"]==case]
        minimum=min(int(r["unique_feasible_layouts"]) for r in selected)
        common=minimum//100*100
        assert common>=100
        norm=bounds[case]
        ideal=np.array(norm["ideal"])
        width=np.array(norm["nadir"])-ideal
        width=np.where(width>0,width,1)
        indicator=HV(ref_point=np.array(norm["reference"]))
        thresholds.append(dict(case=case,smallest_observed_unique=minimum,common_unique_budget=common,
            threshold_rule="floor(min(unique feasible layouts across all 80 runs)/100)*100",
            primary_request_budget=cfg["evaluations"]))
        for row in selected:
            with np.load(src/row["name"]/"evaluations.npz") as archive:
                selected_ids=archive["selected"]
                metrics=archive["metrics"]
                valid=archive["cv"]==0
            valid_indices=np.flatnonzero(valid)
            _,first=np.unique(selected_ids[valid_indices],axis=0,return_index=True)
            first=np.sort(valid_indices[first])
            assert len(first)==int(row["unique_feasible_layouts"])
            truncated=first[:common]
            f=metrics[truncated,:3]*[-1,1,1]
            nd=NDS.do(f,only_non_dominated_front=True)
            matched=float(indicator((f[nd]-ideal)/width))
            full=float(row["HV"])
            assert matched<=full+1e-10
            rows.append(dict(case=case,name=row["name"],variant=row["variant"],seed=row["seed"],
                common_unique_budget=common,requests_to_common_unique=int(truncated[-1])+1,
                final_unique_feasible=len(first),HV_matched_unique=matched,
                HV_full_requests=full,original_request_budget=cfg["evaluations"]))
        per_case=[]
        for variant in cfg["variants"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==variant]
            assert len(rr)==10
            item=dict(case=case,variant=variant,runs=len(rr),common_unique_budget=common)
            for field in ["requests_to_common_unique","final_unique_feasible","HV_matched_unique","HV_full_requests"]:
                value=np.array([r[field] for r in rr],float)
                item[field+"_mean"]=float(value.mean())
                item[field+"_sd"]=float(value.std(ddof=1))
            per_case.append(item)
        rank_full=rankdata([-r["HV_full_requests_mean"] for r in per_case],method="average")
        rank_unique=rankdata([-r["HV_matched_unique_mean"] for r in per_case],method="average")
        for i,r in enumerate(per_case):
            r["rank_full_requests"]=float(rank_full[i])
            r["rank_matched_unique"]=float(rank_unique[i])
        aggregate.extend(per_case)
        print("BUDGET DIAGNOSTIC",case,"common unique",common,flush=True)
    table(out/"per_run_matched_unique.csv",rows)
    table(out/"aggregate_matched_unique.csv",aggregate)
    table(out/"common_unique_thresholds.csv",thresholds)
    return dict(status="complete",runs=len(rows),cases=len(thresholds),new_optimization_runs=0,
        original_primary_tests_unchanged=True,additional_significance_tests=False,
        scope="Posthoc descriptive truncation by a prespecified threshold rule; neither equal wall time nor reoptimization.")
