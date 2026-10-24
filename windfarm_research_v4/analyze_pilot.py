"""All prespecified comparisons, descriptive screening, no significance claims."""
import bootstrap_v4
from pathlib import Path
import json,csv,datetime
import numpy as np
from pymoo.indicators.hv import HV
from algorithms import NDS
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;SRC=HERE/"results/pilot";OUT=SRC/"analysis"
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def table(name,rows):
    with (OUT/name).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def run():
    OUT.mkdir(exist_ok=True);cfg=read(SRC/"frozen_config.json")
    summaries=[read(p) for p in sorted(SRC.glob("*/DONE.json"))];assert len(summaries)==72
    for seed in cfg["seeds"]:
        for case in cfg["cases"]:
            hashes={r["initial_layouts_sha256"] for r in summaries if r["seed"]==seed and r["candidates"]==case["candidates"]}
            assert len(hashes)==1,"Initial phenotype mismatch"
    rows=[];allcurves=[];bounds={}
    for case in cfg["cases"]:
        selected=[r for r in summaries if r["candidates"]==case["candidates"]];data={};pool=[]
        key=f"{case['scenario']}_M{case['candidates']}"
        for r in selected:
            with np.load(SRC/r["name"]/"evaluations.npz") as z:
                f=z["metrics"][:,:3]*[-1,1,1];valid=z["cv"]==0;nd=z["pareto_request_indices"];sel=z["selected"]
            data[r["name"]]=(f,valid,nd,sel);pool.append(f[valid])
        joined=np.vstack(pool);lo=joined.min(0);hi=joined.max(0);width=np.where(hi>lo,hi-lo,1)
        bounds[key]=dict(ideal=lo.tolist(),nadir=hi.tolist(),reference=[1.1]*3)
        hv=HV(ref_point=np.full(3,1.1))
        for r in selected:
            f,valid,nd,sel=data[r["name"]];row=dict(r);row["case"]=key
            row["HV"]=float(hv((f[nd]-lo)/width))
            ui=np.unique(sel,axis=0,return_index=True)[1];ui=ui[valid[ui]]
            row["AEP_LCOE_spearman"]=float(spearmanr(-f[ui,0],f[ui,1]).statistic)
            for i,col in enumerate(["AEP","LCOE","area"]):
                vals=-f[ui,0] if i==0 else f[ui,i]
                row[col+"_min"]=float(vals.min());row[col+"_max"]=float(vals.max())
            rows.append(row)
            archive=np.empty((0,3))
            for t in range(200,cfg["evaluations"]+1,200):
                chunk=f[t-200:t][valid[t-200:t]]
                block=np.unique(np.vstack([archive,chunk]),axis=0)
                archive=block[NDS.do(block,only_non_dominated_front=True)]
                value=float(hv((archive-lo)/width))
                allcurves.append(dict(case=key,variant=r["variant"],seed=r["seed"],requests=t,HV=value))
            assert abs(value-row["HV"])<1e-10
    table("per_run_metrics.csv",rows);table("convergence.csv",allcurves)
    aggregates=[];contrasts=[]
    for case in bounds:
        for v in cfg["variants"]:
            rr=[r for r in rows if r["case"]==case and r["variant"]==v]
            d=dict(case=case,variant=v,runs=len(rr))
            for field in ["HV","duplicate_rate","unique_feasible_layouts","raw_feasible_rate","post_feasible_rate","mean_replacements","mean_population","mean_velocity_nonzero","mean_repair_key_fraction"]:
                d[field+"_mean"]=float(np.mean([r[field] for r in rr]));d[field+"_sd"]=float(np.std([r[field] for r in rr],ddof=1))
            aggregates.append(d)
        for treatment,control in cfg["paired_contrasts"]:
            a={r["seed"]:r for r in rows if r["case"]==case and r["variant"]==treatment}
            b={r["seed"]:r for r in rows if r["case"]==case and r["variant"]==control}
            dif=np.array([a[s]["HV"]-b[s]["HV"] for s in cfg["seeds"]])
            contrasts.append(dict(case=case,treatment=treatment,control=control,seeds=len(dif),
                                  mean_HV_difference=float(dif.mean()),wins=int((dif>1e-12).sum()),ties=int((abs(dif)<=1e-12).sum()),losses=int((dif < -1e-12).sum()),
                                  mean_unique_difference=float(np.mean([a[s]["unique_feasible_layouts"]-b[s]["unique_feasible_layouts"] for s in cfg["seeds"]])),
                                  mean_duplicate_difference=float(np.mean([a[s]["duplicate_rate"]-b[s]["duplicate_rate"] for s in cfg["seeds"]])),
                                  scope="screening only, 3 seeds; no significance claim"))
    table("aggregate_metrics.csv",aggregates);table("paired_contrasts.csv",contrasts)
    (OUT/"normalization.json").write_text(json.dumps(bounds,indent=2),encoding="utf-8")
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
    fig,axs=plt.subplots(1,2,figsize=(14,6),layout="constrained")
    for ax,case in zip(axs,bounds):
        rr=[next(r for r in aggregates if r["case"]==case and r["variant"]==v) for v in cfg["variants"]]
        ax.barh(np.arange(len(rr)),[r["HV_mean"] for r in rr],xerr=[r["HV_sd"] for r in rr],color="#317B91")
        ax.set_yticks(np.arange(len(rr)),cfg["variants"]);ax.invert_yaxis()
        ax.set(title=case+"，3种子筛查",xlabel="共同归一化HV（均值±样本标准差）")
    fig.savefig(OUT/"pilot_HV.png",dpi=180);plt.close(fig)
    (OUT/"DONE.json").write_text(json.dumps(dict(status="complete",runs=72,curves_checked=72,all_initializations_match=True,
                                               scope="prespecified screening, not confirmatory",finished_at=datetime.datetime.now().astimezone().isoformat()),indent=2),encoding="utf-8")
    print("PILOT ANALYSIS COMPLETE",flush=True)
