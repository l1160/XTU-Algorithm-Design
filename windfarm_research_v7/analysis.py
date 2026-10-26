"""Direct paired comparisons, with a predeclared frozen HV scale and CPU traces."""
from pathlib import Path
import csv
import json
import numpy as np
from pymoo.indicators.hv import HV
from windfarm_research_v6.compat import legacy as A
from windfarm_research_v6.lifecycle import save_json,now


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def table(p,rows):
    with Path(p).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(plan):
    output=Path(plan["output"]);folder=output/"analysis";folder.mkdir(exist_ok=True)
    frozen=read(plan["frozen_normalization"])
    low=np.array(frozen["ideal"]);high=np.array(frozen["nadir"]);width=np.array(frozen["width"]);ref=np.array(frozen["reference"])
    indicator=HV(ref_point=ref)
    data=[]
    for job in plan["jobs"]:
        src=Path(job["output"]);done=read(src/"DONE.json")
        assert done["status"]=="complete" and done["requests"]==10000
        with np.load(src/"evaluations.npz",allow_pickle=False) as z:
            metrics=z["metrics"].copy();valid=z["cv"]==0;front=z["pareto_request_indices"].copy()
        assert valid[front].all()
        data.append(dict(method=job["method"],done=done,metrics=metrics,valid=valid,front=front,
                         f=metrics[:,:3]*[-1,1,1],trace=read(src/"trace.json")))
    assert len({x["done"]["initial_layouts_sha256"] for x in data})==1
    assert all(np.array_equal(x["metrics"][:100],data[0]["metrics"][:100]) for x in data[1:])
    pool=np.vstack([x["f"][x["valid"]] for x in data])
    pooled_low=np.minimum(low,pool.min(0));pooled_high=np.maximum(high,pool.max(0));pooled_width=np.maximum(pooled_high-pooled_low,1e-12)
    save_json(folder/"normalization.json",dict(primary=frozen,
        primary_scope="Frozen before V7 runs, from the completed V6 comparison; negative normalized improvements are allowed; points outside reference contribute zero",
        secondary=dict(ideal=pooled_low.tolist(),nadir=pooled_high.tolist(),width=pooled_width.tolist(),reference=ref.tolist()),
        secondary_scope="Joint envelope of previous V6 data and all four V7 runs"))

    def value(points,lo=low,span=width):
        z=np.unique((points-lo)/span,axis=0);z=z[np.all(z<ref,axis=1)]
        return float(indicator(z)) if len(z) else 0.

    rows=[];curves=[];attainment={}
    final_values={d["method"]:value(d["f"][d["front"]]) for d in data}
    for d in data:
        done=d["done"];method=d["method"]
        row=dict(method=method,HV=final_values[method],pooled_HV=value(d["f"][d["front"]],pooled_low,pooled_width),
            worker_cpu_seconds=done["worker_cpu_seconds"],worker_wall_seconds=done["worker_wall_seconds"],
            feasible_rate=done["post_feasible_rate"],unique_feasible_layouts=done["unique_feasible_layouts"],
            duplicate_request_rate=done["duplicate_request_rate"],pareto_layouts=len(d["front"]),
            mean_working_population=float(np.mean([x["working_population"] for x in d["trace"]])),
            native_operator_requests=done["operator_counts"].get("native_operator_requests",0),
            local_exact_requests=done["operator_counts"].get("local_exact_requests",0))
        for key,i in (("AEP_GWh",0),("LCOE_USD2022_MWh",1),("area_km2",2)):
            row[key+"_min"]=float(d["metrics"][d["front"],i].min());row[key+"_max"]=float(d["metrics"][d["front"],i].max())
        rows.append(row)
        current=np.empty((0,3));previous=0;target=final_values[method.removesuffix("_problem")]
        for point in d["trace"]:
            count=point["evaluations"]
            block=np.unique(np.vstack([current,d["f"][previous:count][d["valid"][previous:count]]]),axis=0)
            current=block[A.NDS.do(block,only_non_dominated_front=True)]
            hv=value(current);previous=count
            curves.append(dict(method=method,requests=count,HV=hv,cpu_seconds=point["elapsed_cpu_seconds"],wall_seconds=point["elapsed_wall_seconds"]))
            if method not in attainment and hv>=target-1e-10:
                attainment[method]=dict(requests=count,cpu_seconds=point["elapsed_cpu_seconds"],wall_seconds=point["elapsed_wall_seconds"])
        assert abs(value(current)-row["HV"])<1e-9
    differences=[]
    for base in ("LMOCSO","DGEA"):
        a=next(r for r in rows if r["method"]==base);b=next(r for r in rows if r["method"]==base+"_problem")
        hit=attainment.get(base+"_problem")
        differences.append(dict(baseline=base,HV_change_percent=100*(b["HV"]/a["HV"]-1),
             pooled_HV_change_percent=100*(b["pooled_HV"]/a["pooled_HV"]-1),
             CPU_change_percent=100*(b["worker_cpu_seconds"]/a["worker_cpu_seconds"]-1),
             baseline_quality_reached_at_requests=hit["requests"] if hit else None,
             baseline_quality_reached_at_CPU_seconds=hit["cpu_seconds"] if hit else None,
             CPU_fraction_to_reach_baseline_final_HV=hit["cpu_seconds"]/a["worker_cpu_seconds"] if hit else None,
             scope="One development seed; descriptive only. Both versions use the same exact accelerated evaluator."))
    table(folder/"summary.csv",rows);table(folder/"paired_differences.csv",differences);table(folder/"convergence.csv",curves)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for d in data:
        cc=[r for r in curves if r["method"]==d["method"]]
        name=d["method"]+(' V7' if d["method"].endswith('_problem') else ' 基线')
        axes[0].plot([r["requests"] for r in cc],[r["HV"] for r in cc],label=name)
        axes[1].plot([r["cpu_seconds"] for r in cc],[r["HV"] for r in cc],label=name)
    axes[0].set(xlabel="评价请求数",ylabel="HV（冻结边界）",title="同预算解集质量")
    axes[1].set(xlabel="工作进程CPU秒",ylabel="HV（冻结边界）",title="质量与计算时间")
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("LMOCSO / DGEA：单一集成改进版本的直接对照，种子901")
    fig.savefig(folder/"comparison.png",dpi=150);plt.close(fig)
    result=dict(status="complete",finished_at=now(),runs=4,shared_initialization_verified=True,
                differences=differences,scope="Development comparison, not independent multi-seed confirmation")
    save_json(folder/"DONE.json",result)
    return result
