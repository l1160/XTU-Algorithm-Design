"""One authorized six-method paired comparison, sequential and restartable."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
EXPECTED=["CCGDE3","CCGDE3_problem","LMOCSO","LMOCSO_problem","DGEA","DGEA_problem"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def configure(plan):
    sys.path.insert(0,str(ROOT))
    if plan.get("dependency_path"):
        sys.path.insert(0,plan["dependency_path"])
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
                      MPLBACKEND="Agg",WINDFARM_V6_BACKGROUND="1")


def verify(plan):
    if [job["method"] for job in plan["jobs"]]!=EXPECTED:
        raise ValueError("This controller is restricted to the declared three paired comparisons")
    for path,expected in plan["source_sha256"].items():
        if digest(ROOT/path)!=expected:
            raise ValueError(f"Frozen source changed: {path}; preserve the existing run")
    for job in plan["jobs"]:
        if digest(job["config"])!=job["config_sha256"]:
            raise ValueError(f"Frozen configuration changed: {job['method']}")
        expected_job={"mode":"Run","config":job["config"],"output":job["output"],
                      "dependency_path":plan["dependency_path"]}
        if read(job["worker_job"])!=expected_job:
            raise ValueError(f"Worker routing differs from the plan: {job['method']}")
    from windfarm_research_v6.run import validate_config
    configs=[validate_config(read(job["config"])) for job in plan["jobs"]]
    shared=[{k:v for k,v in cfg.items() if k!="method"} for cfg in configs]
    if any(c!=shared[0] for c in shared[1:]):
        raise ValueError("Comparison cells do not share the same case, seed and budget")


def prepare(plan):
    from windfarm_research_v6.model import OverlapFarm
    from windfarm_research_v6.lifecycle import save_json,now
    from windfarm_research_v6.run import validate_config
    import numpy as np
    output=Path(plan["output"])
    cfg=validate_config(read(plan["jobs"][0]["config"]))
    started,cpu=time.perf_counter(),time.process_time()
    print(now(),"COMMON SETUP START",cfg["candidates"],cfg["turbines"],flush=True)
    farm=OverlapFarm(cfg["scenario"],cfg["candidates"],cfg["turbines"],cfg["candidate_seed"],
                     terrain_kind=cfg["terrain"],initial_size=cfg["population"])
    farm.ensure_wake();farm.ensure_ground()
    assert np.isfinite(farm.wake).all() and len(farm.records)==0
    assert len(farm.certificate)>=farm.n
    save_json(output/"common_setup.json",dict(status="complete",finished_at=now(),
              wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
              optimization_requests=0,case=farm.case_metadata(),
              scope="Shared read-only wake and terrain-distance caches prepared before all six methods; report separately from solver time"))
    print(now(),"COMMON SETUP DONE",flush=True)


def table(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def analyze(plan):
    import numpy as np
    from pymoo.indicators.hv import HV
    from windfarm_research_v6.compat import legacy as A
    from windfarm_research_v6.lifecycle import save_json,now
    from windfarm_research_v6.run import signature,validate_config
    output=Path(plan["output"]);folder=output/"analysis";folder.mkdir(exist_ok=True)
    data=[]
    for job in plan["jobs"]:
        source=Path(job["output"]);done=read(source/"DONE.json")
        cfg=validate_config(read(job["config"]))
        assert read(source/"manifest.json")==signature(cfg)
        assert done["status"]=="complete" and done["requests"]==cfg["requests"]
        with np.load(source/"evaluations.npz",allow_pickle=False) as z:
            metrics=z["metrics"].copy();valid=z["cv"]==0;indices=z["pareto_request_indices"].copy()
        assert valid[indices].all()
        data.append(dict(job=job,done=done,metrics=metrics,valid=valid,front=indices,
                         f=metrics[:,:3]*[-1,1,1],trace=read(source/"trace.json")))
    assert len({d["done"]["initial_layouts_sha256"] for d in data})==1
    assert len({d["done"]["initial_keys_sha256"] for d in data})==1
    initial=plan["population"]
    assert all(np.array_equal(d["metrics"][:initial],data[0]["metrics"][:initial]) for d in data[1:])
    merged=np.vstack([d["f"][d["valid"]] for d in data])
    low,high=merged.min(axis=0),merged.max(axis=0)
    width=np.where(high>low,high-low,1.)
    hv=HV(ref_point=np.full(3,1.1))
    save_json(folder/"normalization.json",dict(ideal=low.tolist(),nadir=high.tolist(),width=width.tolist(),reference=[1.1]*3,
              zero_range_axes=np.flatnonzero(high==low).tolist(),scope="Common pooled empirical bounds of all feasible visited layouts in these six runs; HV is not a percentage"))
    rows=[];curves=[];front_rows=[]
    for d in data:
        done=d["done"];method=d["job"]["method"]
        value=float(hv(np.unique((d["f"][d["front"]]-low)/width,axis=0)))
        counts=done["operator_counts"]
        proposals=counts.get("offspring_proposals",0)
        row=dict(method=method,seed=done["config"]["seed"],requests=done["requests"],HV=value,
                 feasible_rate=done["post_feasible_rate"],unique_feasible_layouts=done["unique_feasible_layouts"],
                 duplicate_request_rate=done["duplicate_request_rate"],
                 offspring_primary_duplicate_rate=counts.get("offspring_primary_duplicate_proposals",0)/proposals if proposals else "",
                 mean_recorded_population=float(np.mean([x["population"] for x in d["trace"]])),
                 mean_recorded_physical_unique=float(np.mean([x["physical_unique"] for x in d["trace"]])),
                 worker_wall_seconds=done["worker_wall_seconds"],worker_cpu_seconds=done["worker_cpu_seconds"],
                 initialization_wall_seconds=done["initialization_wall_seconds"],restore_wall_seconds=done["restore_wall_seconds"],
                 new_feasible_evaluations=done["statistics"]["new_feasible_evaluations"],
                 full_wake_evaluations=done["statistics"].get("full_wake_evaluations",0),
                 incremental_wake_evaluations=done["statistics"].get("incremental_wake_evaluations",0),
                 pareto_layouts=len(d["front"]))
        for field,col in (("AEP_GWh",0),("LCOE_USD2022_MWh",1),("effective_area_km2",2)):
            row[field+"_min"]=float(d["metrics"][d["front"],col].min())
            row[field+"_max"]=float(d["metrics"][d["front"],col].max())
        rows.append(row)
        front=np.empty((0,3));previous=0
        checkpoints=sorted(set([initial]+list(range(500,len(d["f"])+1,500))+[len(d["f"])]))
        for count in checkpoints:
            block=np.unique(np.vstack([front,d["f"][previous:count][d["valid"][previous:count]]]),axis=0)
            front=block[A.NDS.do(block,only_non_dominated_front=True)]
            curves.append(dict(method=method,requests=count,HV=float(hv((front-low)/width))))
            previous=count
        assert abs(curves[-1]["HV"]-value)<1e-9
        for i in d["front"]:
            front_rows.append(dict(method=method,request_index_zero_based=int(i),
                                   AEP_GWh=float(d["metrics"][i,0]),LCOE_USD2022_MWh=float(d["metrics"][i,1]),
                                   effective_area_km2=float(d["metrics"][i,2])))
    differences=[]
    for base in ("CCGDE3","LMOCSO","DGEA"):
        a=next(r for r in rows if r["method"]==base)
        b=next(r for r in rows if r["method"]==base+"_problem")
        differences.append(dict(baseline=base,improved=base+"_problem",HV_difference=b["HV"]-a["HV"],
                                unique_feasible_difference=b["unique_feasible_layouts"]-a["unique_feasible_layouts"],
                                CPU_ratio_improved_over_baseline=b["worker_cpu_seconds"]/a["worker_cpu_seconds"],
                                wall_ratio_improved_over_baseline=b["worker_wall_seconds"]/a["worker_wall_seconds"],
                                scope="One paired seed; descriptive only, no significance or general superiority claim"))
    table(folder/"summary.csv",rows);table(folder/"paired_differences.csv",differences)
    table(folder/"convergence.csv",curves);table(folder/"pareto_objectives.csv",front_rows)
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout="constrained")
    names=[r["method"] for r in rows];x=np.arange(6);colors=["#315C8A","#D68732"]*3
    axes[0].bar(x,[r["HV"] for r in rows],color=colors)
    axes[0].set_title("解集质量（同一归一化）");axes[0].set_ylabel("HV，非百分比")
    axes[0].set_xticks(x,names,rotation=55,ha="right")
    for method in names:
        points=[r for r in curves if r["method"]==method]
        axes[1].plot([r["requests"] for r in points],[r["HV"] for r in points],label=method)
    axes[1].set(xlabel="评价请求数",ylabel="HV",title="收敛过程")
    axes[1].legend(fontsize=7)
    axes[2].bar(x,[r["worker_cpu_seconds"] for r in rows],color=colors)
    axes[2].set_title("工作进程CPU时间");axes[2].set_ylabel("秒")
    axes[2].set_xticks(x,names,rotation=55,ha="right")
    fig.suptitle("六方法首轮配对比较：单场景、单种子，不作显著性结论")
    fig.savefig(folder/"comparison.png",dpi=160);plt.close(fig)
    note=("六方法首轮配对对照已完成。\n\n"
          "场景obs_00，flat，1600候选点，400台风机，种群初始100，每次10000请求，种子901。\n"
          "六方法初始机位、随机键及目标值完全相同，使用同一V6转子重叠模型。\n"
          "summary.csv给出全部方法结果；paired_differences.csv给出三组差值；convergence.csv与comparison.png给出收敛和比较。\n"
          "只有一个配对种子，结果为描述性证据，不能宣称稳定优于基线或完成结项。HV不是百分数。\n"
          "所有方法顺序执行，通用缓存提前准备并另记时间。墙钟时间仍可能受系统负载及暂停影响；CPU时间、初始化和恢复应同时审阅。\n"
          "CCGDE3沿用仓库的CCGDE3-inspired尾流分组适配，不声称严格复现原论文。\n")
    (folder/"结果说明.txt").write_text(note,encoding="utf-8")
    save_json(folder/"DONE.json",dict(status="complete",runs=6,paired_seeds=1,shared_initialization_verified=True,
              convergence_endpoints_verified=True,finished_at=now(),differences=differences))
    print(now(),"ANALYSIS COMPLETE",flush=True)


def campaign(plan,plan_path):
    from windfarm_research_v6.lifecycle import FileLock,save_json,now
    from windfarm_research_v6.run import signature,validate_config
    output=Path(plan["output"]);output.mkdir(parents=True,exist_ok=True)
    plan_sha=digest(plan_path)
    with FileLock(output/"campaign.lock"):
        if (output/"DONE.json").exists():
            assert read(output/"DONE.json")["plan_sha256"]==plan_sha
            print(now(),"CAMPAIGN ALREADY COMPLETE",flush=True);return
        completed=[]
        current=None
        state=dict(status="starting",plan_sha256=plan_sha,total_runs=6,completed_methods=completed,
                   active_method=None,updated_at=now(),controller_pid=os.getpid())
        keep_awake=False
        try:
            if os.name=="nt":
                import ctypes
                keep_awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            state["automatic_sleep_inhibition_requested"]=keep_awake

            def publish(status,method=None):
                state.update(status=status,active_method=method,updated_at=now(),completed_methods=completed.copy())
                if current and (Path(current["output"])/"progress.json").exists():
                    state["active_progress"]=read(Path(current["output"])/"progress.json")
                else:state.pop("active_progress",None)
                save_json(output/"progress.json",state)

            def child(command,label,job=None):
                stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                logs=output/"controller_logs";logs.mkdir(exist_ok=True)
                with (logs/(label+"_"+stamp+".log")).open("a",encoding="utf-8",buffering=1) as stream:
                    kwargs=dict(cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    if os.name=="nt":kwargs["creationflags"]=subprocess.CREATE_NO_WINDOW
                    process=subprocess.Popen(command,**kwargs)
                    state["child_pid"]=process.pid
                    while process.poll() is None:
                        if job and (output/"PAUSE_REQUEST.json").exists():
                            pause=Path(job["output"])/"PAUSE_REQUEST.json"
                            if not pause.exists():save_json(pause,{"controller_plan":plan_sha})
                        publish(state["status"],state["active_method"])
                        time.sleep(3)
                    if process.returncode:
                        raise RuntimeError(f"{label} exited {process.returncode}; see its new logs, preserve full checkpoints")
                    state.pop("child_pid",None)

            python=str(Path(sys.executable).with_name("python.exe"))
            if not (output/"common_setup.json").exists():
                publish("preparing_common_model")
                child([python,"-u",str(Path(__file__).resolve()),"--plan",str(plan_path),"--stage","prepare"],"prepare")
            for job in plan["jobs"]:
                current=job
                verify(plan)
                cfg=validate_config(read(job["config"]))
                directory=Path(job["output"])
                if (directory/"DONE.json").exists():
                    assert read(directory/"manifest.json")==signature(cfg)
                    assert read(directory/"DONE.json")["requests"]==cfg["requests"]
                    completed.append(job["method"]);continue
                if (output/"PAUSE_REQUEST.json").exists():
                    publish("paused");return
                pause=directory/"PAUSE_REQUEST.json"
                if pause.exists() and read(pause).get("controller_plan")==plan_sha:
                    pause.unlink()
                publish("running",job["method"])
                print(now(),"RUN",job["method"],flush=True)
                child([python,"-u",str(ROOT/"windfarm_research_v6/worker.py"),"--job",job["worker_job"]],job["method"],job)
                if not (directory/"DONE.json").exists():
                    if (directory/"progress.json").exists() and read(directory/"progress.json")["status"]=="paused":
                        publish("paused",job["method"]);return
                    raise RuntimeError(f"Worker exited without completed results: {job['method']}")
                completed.append(job["method"])
                print(now(),"RUN COMPLETE",job["method"],flush=True)
            current=None
            publish("analyzing")
            child([python,"-u",str(Path(__file__).resolve()),"--plan",str(plan_path),"--stage","analyze"],"analysis")
            result=read(output/"analysis/DONE.json")
            save_json(output/"DONE.json",dict(result,plan_sha256=plan_sha,methods=completed))
            publish("complete")
            print(now(),"CAMPAIGN COMPLETE",flush=True)
        except BaseException:
            state.update(status="failed",error=traceback.format_exc(),updated_at=now())
            save_json(output/"FAILED.json",state);save_json(output/"progress.json",state)
            raise
        finally:
            if keep_awake:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--stage",choices=("campaign","prepare","analyze","inspect"),default="campaign")
    args=parser.parse_args();plan_path=Path(args.plan).resolve();plan=read(plan_path)
    output=Path(plan["output"]);output.mkdir(parents=True,exist_ok=True)
    if args.stage=="campaign":
        stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sys.stdin=open(os.devnull)
        sys.stdout=sys.stderr=(output/("campaign_"+stamp+".log")).open("a",encoding="utf-8",buffering=1)
        try:
            configure(plan);verify(plan)
            campaign(plan,plan_path)
        except BaseException:traceback.print_exc();raise
    else:
        configure(plan);verify(plan)
        if args.stage=="prepare":prepare(plan)
        elif args.stage=="analyze":analyze(plan)
        else:print(json.dumps({"validated":True,"methods":EXPECTED,"runs":6,"workers":1,"optimization_started":False},ensure_ascii=False))


if __name__=="__main__":main()
