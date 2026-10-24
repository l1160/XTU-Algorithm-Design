"""Per-terrain screening statistics; no cross-terrain HV or significance claim."""
import csv,json,datetime
import numpy as np
from algorithms import NDS
from pymoo.indicators.hv import HV
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def table(p,rows):
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def run(src,cfg):
    out=src/"analysis";out.mkdir(exist_ok=True)
    summaries=[read(p) for p in src.glob("*/DONE.json") if (p.parent/"evaluations.npz").exists()]
    assert len(summaries)==90
    rows=[];curves=[];bounds={}
    for c in cfg["cases"]:
        selected=[r for r in summaries if r["terrain"]==c["terrain"] and r["candidates"]==c["candidates"]
                  and r["scenario"]==c["scenario"] and r["turbines"]==c["turbines"]]
        key=f"{c['scenario']}_{c['terrain']}_M{c['candidates']}_N{c['turbines']}"
        for s in cfg["seeds"]:assert len({r["initial_layouts_sha256"] for r in selected if r["seed"]==s})==1
        data={};pool=[]
        for r in selected:
            with np.load(src/r["name"]/"evaluations.npz") as z:
                metric=z["metrics"];valid=z["cv"]==0;nd=z["pareto_request_indices"]
            f=metric[:,:3]*[-1,1,1];data[r["name"]]=(f,valid,nd,metric);pool.append(f[valid])
        allf=np.vstack(pool);lo=allf.min(0);hi=allf.max(0);width=np.where(hi>lo,hi-lo,1)
        bounds[key]=dict(ideal=lo.tolist(),nadir=hi.tolist(),reference=[1.1]*3)
        hv=HV(ref_point=np.full(3,1.1))
        for r in selected:
            f,valid,nd,m=data[r["name"]];row=dict(r,case=key,HV=float(hv((f[nd]-lo)/width)))
            for col,i in [("AEP",0),("LCOE",1),("effective_area",2),("surface_MST",3),("gross_area",4)]:
                row[col+"_min"]=float(m[valid,i].min());row[col+"_max"]=float(m[valid,i].max())
            rows.append(row);front=np.empty((0,3))
            for t in range(200,cfg["evaluations"]+1,200):
                block=np.unique(np.vstack([front,f[t-200:t][valid[t-200:t]]]),axis=0)
                front=block[NDS.do(block,only_non_dominated_front=True)]
                score=float(hv((front-lo)/width))
                curves.append(dict(case=key,variant=r["variant"],seed=r["seed"],requests=t,HV=score))
            assert abs(score-row["HV"])<1e-10
    aggregate=[]
    for case in bounds:
        for v in cfg["variants"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==v]
            d=dict(case=case,variant=v,n=len(rr))
            for field in ["HV","unique_feasible_layouts","duplicate_rate","post_feasible_rate","raw_feasible_rate","mean_population","elapsed_seconds"]:
                d[field+"_mean"]=float(np.mean([r[field] for r in rr]));d[field+"_sd"]=float(np.std([r[field] for r in rr],ddof=1))
            aggregate.append(d)
    table(out/"per_run_metrics.csv",rows);table(out/"aggregate_metrics.csv",aggregate);table(out/"convergence.csv",curves)
    (out/"normalization.json").write_text(json.dumps(bounds,indent=2),encoding="utf-8")
    (out/"DONE.json").write_text(json.dumps(dict(status="complete",runs=90,all_initializations_matched=True,
        convergence_endpoints_verified=90,scope=cfg["scope"],finished_at=datetime.datetime.now().astimezone().isoformat()),indent=2),encoding="utf-8")
    print("TERRAIN ANALYSIS COMPLETE",flush=True)
