"""Two-wind ridge confirmation: pooled within-case HV and six paired contrasts."""
import json
import csv
import itertools
import datetime
import numpy as np
from algorithms import NDS
from pymoo.indicators.hv import HV


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(src, cfg):
    out=src/"analysis";out.mkdir(exist_ok=True)
    summaries=[json.loads(p.read_text()) for p in src.glob("*/DONE.json") if (p.parent/"evaluations.npz").exists()]
    assert len(summaries)==60
    rows=[];curves=[];bounds={};comparisons=[];pairs=[]
    for c in cfg["cases"]:
        name=f"{c['scenario']}_{c['terrain']}_M{c['candidates']}_N{c['turbines']}"
        rr=[r for r in summaries if all(r[k]==c[k] for k in ("scenario","terrain","candidates","turbines"))]
        assert len(rr)==30
        for seed in cfg["seeds"]:
            group=[r for r in rr if r["seed"]==seed]
            assert len(group)==3
            for field in ("initial_keys_sha256","initial_layouts_sha256","initial_metrics_sha256"):
                assert len({r[field] for r in group})==1,field
        data={};pool=[]
        for r in rr:
            with np.load(src/r["name"]/"evaluations.npz") as z:
                metrics=z["metrics"].copy();valid=z["cv"]==0;nd=z["pareto_request_indices"].copy()
            f=metrics[:,:3]*[-1,1,1]
            assert len(f)==10000 and valid[nd].all()
            data[r["name"]]=(metrics,f,valid,nd);pool.append(f[valid])
        merged=np.vstack(pool);lo=merged.min(0);hi=merged.max(0);width=np.where(hi>lo,hi-lo,1.)
        bounds[name]=dict(ideal=lo.tolist(),nadir=hi.tolist(),width=width.tolist(),reference=[1.1]*3,
                         zero_range_axes=np.flatnonzero(hi==lo).tolist())
        hv=HV(ref_point=np.full(3,1.1))
        for r in rr:
            metrics,f,valid,nd=data[r["name"]]
            row=dict(r,case=name,HV=float(hv((f[nd]-lo)/width)))
            for label,i in (("AEP_GWh",0),("LCOE_USD2022_MWh",1),("effective_area_km2",2),("surface_MST_m",3),("gross_area_km2",4)):
                row[label+"_min"]=float(metrics[valid,i].min());row[label+"_max"]=float(metrics[valid,i].max())
            rows.append(row)
            front=np.empty((0,3))
            for t in range(500,10001,500):
                block=np.unique(np.vstack([front,f[t-500:t][valid[t-500:t]]]),axis=0)
                front=block[NDS.do(block,only_non_dominated_front=True)]
                value=float(hv((front-lo)/width))
                curves.append(dict(case=name,variant=r["variant"],seed=r["seed"],requests=t,HV=value))
            assert abs(value-row["HV"])<1e-10
        for treatment,control in cfg["paired_contrasts"]:
            a={r["seed"]:r["HV"] for r in rows if r["case"]==name and r["variant"]==treatment}
            b={r["seed"]:r["HV"] for r in rows if r["case"]==name and r["variant"]==control}
            delta=np.array([a[s]-b[s] for s in cfg["seeds"]])
            signs=np.array(list(itertools.product([-1,1],repeat=10)))
            p=float(np.mean(abs(signs@delta/10)>=abs(delta.mean())-1e-14))
            sums=np.array([0.])
            for d in delta:sums=np.r_[sums+d,sums-d]
            assert p==float(np.mean(abs(sums/10)>=abs(delta.mean())-1e-14))
            comparisons.append(dict(case=name,treatment=treatment,control=control,n=10,
                mean_HV_difference=float(delta.mean()),median_HV_difference=float(np.median(delta)),
                wins=int((delta>1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()),losses=int((delta< -1e-12).sum()),paired_signflip_p=p))
            pairs.extend(dict(case=name,seed=s,treatment=treatment,control=control,treatment_HV=a[s],control_HV=b[s],difference=a[s]-b[s]) for s in cfg["seeds"])
    assert len(comparisons)==6
    order=np.argsort([r["paired_signflip_p"] for r in comparisons]);last=0.
    for rank,i in enumerate(order):
        last=max(last,min(1.,comparisons[i]["paired_signflip_p"]*(6-rank)));comparisons[i]["Holm_adjusted_p"]=last
    aggregate=[]
    for case in bounds:
        for variant in cfg["variants"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==variant]
            item=dict(case=case,variant=variant,n=len(rr))
            for field in ("HV","unique_feasible_layouts","duplicate_rate","post_feasible_rate","raw_feasible_rate","mean_population","elapsed_seconds","initialization_seconds"):
                item[field+"_mean"]=float(np.mean([r[field] for r in rr]));item[field+"_sd"]=float(np.std([r[field] for r in rr],ddof=1))
            aggregate.append(item)
    table(out/"per_run_metrics.csv",rows);table(out/"aggregate_metrics.csv",aggregate)
    table(out/"convergence.csv",curves);table(out/"paired_comparisons.csv",comparisons);table(out/"individual_paired_differences.csv",pairs)
    (out/"normalization.json").write_text(json.dumps(bounds,indent=2),encoding="utf-8")
    (out/"DONE.json").write_text(json.dumps(dict(status="complete",runs=60,all_initial_keys_layouts_metrics_matched=True,
        all_convergence_endpoints_matched=True,family_size=6,independent_signflip_enumeration_passed=True,
        scope="Synthetic ridge M1600/N400 under two imposed wind distributions; declared constrained adaptations only; no real-site validation",
        assumptions="Paired differences sign-exchangeable under null; independent optimization seeds. Shared normalization is empirical and case-specific.",
        finished_at=datetime.datetime.now().astimezone().isoformat()),indent=2),encoding="utf-8")
