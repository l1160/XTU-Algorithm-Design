"""Confirmation metrics and prespecified paired comparisons, all cases retained."""
import csv,itertools,json,datetime
import numpy as np
from pymoo.indicators.hv import HV
from algorithms import NDS

def read(p):return json.loads(p.read_text(encoding="utf-8"))
def table(p,rows):
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def signflip_p(d):
    d=np.asarray(d,float)
    assert len(d)<=16
    observed=abs(d.mean())
    signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
    return float(np.mean(np.abs(signs@d/len(d))>=observed-1e-14))
def run(src,cfg):
    out=src/"analysis";out.mkdir(exist_ok=True)
    expected=len(cfg["cases"])*len(cfg["variants"])*len(cfg["seeds"])
    summaries=[read(p) for p in sorted(src.glob("*/DONE.json")) if (p.parent/"evaluations.npz").exists()]
    assert len(summaries)==expected
    rows=[];curves=[];bounds={};contrasts=[]
    for case in cfg["cases"]:
        rr=[r for r in summaries if r["scenario"]==case["scenario"] and r["candidates"]==case["candidates"] and r["turbines"]==case["turbines"]]
        name=f"{case['scenario']}_M{case['candidates']}_N{case['turbines']}"
        for seed in cfg["seeds"]:
            hashes={r["initial_layouts_sha256"] for r in rr if r["seed"]==seed}
            assert len(hashes)==1,"Initial layouts must match across paired methods"
        data={};pool=[]
        for r in rr:
            with np.load(src/r["name"]/"evaluations.npz") as z:
                f=z["metrics"][:,:3]*[-1,1,1];valid=z["cv"]==0;nd=z["pareto_request_indices"]
            assert len(f)==cfg["evaluations"]
            data[r["name"]]=(f,valid,nd);pool.append(f[valid])
        joined=np.vstack(pool);lo=joined.min(0);hi=joined.max(0);width=np.where(hi>lo,hi-lo,1)
        hv=HV(ref_point=np.full(3,1.1))
        bounds[name]=dict(ideal=lo.tolist(),nadir=hi.tolist(),reference=[1.1]*3)
        for r in rr:
            f,valid,nd=data[r["name"]];row=dict(r,case=name,HV=float(hv((f[nd]-lo)/width)))
            rows.append(row);front=np.empty((0,3))
            for t in range(500,cfg["evaluations"]+1,500):
                block=np.unique(np.vstack([front,f[t-500:t][valid[t-500:t]]]),axis=0)
                front=block[NDS.do(block,only_non_dominated_front=True)]
                value=float(hv((front-lo)/width))
                curves.append(dict(case=name,variant=r["variant"],seed=r["seed"],requests=t,HV=value))
            assert abs(value-row["HV"])<1e-10
        for treatment,control in cfg["paired_contrasts"]:
            a={r["seed"]:r for r in rows if r["case"]==name and r["variant"]==treatment}
            b={r["seed"]:r for r in rows if r["case"]==name and r["variant"]==control}
            d=np.array([a[s]["HV"]-b[s]["HV"] for s in cfg["seeds"]])
            contrasts.append(dict(case=name,treatment=treatment,control=control,n=len(d),
                mean_HV_difference=float(d.mean()),median_HV_difference=float(np.median(d)),
                wins=int((d>1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()),
                paired_signflip_p=signflip_p(d)))
    order=np.argsort([r["paired_signflip_p"] for r in contrasts]);last=0.
    for rank,index in enumerate(order):
        last=max(last,min(1.,contrasts[index]["paired_signflip_p"]*(len(order)-rank)))
        contrasts[index]["Holm_adjusted_p"]=last
    aggregate=[]
    for case in bounds:
        for variant in cfg["variants"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==variant]
            d=dict(case=case,variant=variant,n=len(rr))
            for col in ["HV","unique_feasible_layouts","duplicate_rate","post_feasible_rate","mean_population","elapsed_seconds"]:
                vals=[r[col] for r in rr];d[col+"_mean"]=float(np.mean(vals));d[col+"_sd"]=float(np.std(vals,ddof=1))
            aggregate.append(d)
    table(out/"per_run_metrics.csv",rows);table(out/"convergence.csv",curves)
    table(out/"aggregate_metrics.csv",aggregate);table(out/"paired_comparisons.csv",contrasts)
    (out/"normalization.json").write_text(json.dumps(bounds,indent=2),encoding="utf-8")
    (out/"DONE.json").write_text(json.dumps(dict(status="complete",runs=expected,
        all_initializations_matched=True,all_convergence_endpoints_matched=True,
        family_size=len(contrasts),test="Two-sided exact paired sign-flip test of mean difference, Holm across all prespecified pairs and cases",
        assumptions="Paired differences sign-exchangeable under the null; independent seeds. Empirical normalization remains study-specific.",
        finished_at=datetime.datetime.now().astimezone().isoformat()),indent=2),encoding="utf-8")
    print("CONFIRMATION ANALYSIS COMPLETE",flush=True)
