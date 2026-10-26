"""Detached, serial, resumable confirmation. No automatic extra experiments."""
from pathlib import Path
import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def warm(plan):
    import numpy as np
    from windfarm_research_v7.model import FastFarm
    from windfarm_research_v6.lifecycle import now, sha
    from windfarm_validation.runtime import save_json
    output=Path(plan["output"])
    for case in plan["cases"]:
        folder=output/"preparation"/case["id"]
        folder.mkdir(parents=True,exist_ok=True)
        if (folder/"DONE.json").exists():
            value=read(folder/"DONE.json")
            assert value["geometry_sha256"]==sha(folder/"geometry.npz")
            continue
        save_json(output/"progress.json",dict(status="preparing_shared_caches",active_case=case["id"],
            total_runs=len(plan["jobs"]),completed_runs=0,updated_at=now()))
        started=time.perf_counter()
        print(now(),"PREPARE",case["id"],flush=True)
        farm=FastFarm(case["scenario"],case["candidates"],case["turbines"],20261012,terrain_kind=case["terrain"])
        assert len(farm.certificate)>=case["turbines"] and farm.feasible(farm.certificate[:case["turbines"]])
        farm.ensure_wake()
        farm.ensure_ground()
        np.savez_compressed(folder/"geometry.npz",candidate_xy=farm.xy,elevation=farm.terrain.z,
                            forbidden=farm.terrain.forbidden_cells,certificate=farm.certificate)
        metadata=farm.case_metadata()
        metadata.update(minimum_spacing_m=farm.scene.min_spacing,rated_kw=farm.scene.rated_kw,
                        candidate_count=farm.m,turbine_count=farm.n)
        save_json(folder/"metadata.json",metadata)
        save_json(folder/"DONE.json",dict(status="complete",case=case["id"],wall_seconds=time.perf_counter()-started,
            geometry_sha256=sha(folder/"geometry.npz"),feasible_certificate_size=len(farm.certificate),
            optimization_requests=0,finished_at=now()))
        print(now(),"PREPARED",case["id"],"certificate",len(farm.certificate),flush=True)


def campaign(job):
    from windfarm_research_v6.lifecycle import FileLock, now, sha
    from windfarm_research_v6.compat import ROOT
    from windfarm_validation.runtime import save_json, signature
    assert sha(job["plan"])==job["plan_sha256"],"Frozen plan changed"
    plan=read(job["plan"])
    output=Path(plan["output"])
    assert str(output)==job["output"] and plan["workers"]==1 and len(plan["jobs"])==180
    assert sha(Path(plan["check_output"])/"DONE.json")==plan["check_sha256"]
    def verify_sources():
        for path,digest in plan["source_sha256"].items():assert sha(ROOT/path)==digest,path
    verify_sources()
    with FileLock(output/"campaign.lock"):
        if (output/"DONE.json").exists():return
        inhibited=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
        completed=[]
        durations=[]
        try:
            warm(plan)
            for cell in plan["jobs"]:
                verify_sources()
                assert sha(cell["config"])==cell["config_sha256"]
                assert sha(cell["worker_job"])==cell["worker_job_sha256"]
                config=read(cell["config"])
                folder=Path(cell["output"])
                if (folder/"DONE.json").exists():
                    assert read(folder/"manifest.json")==signature(config)
                    done=read(folder/"DONE.json")
                    assert done["status"]=="complete" and done["requests"]==10000
                    completed.append(cell["id"])
                    durations.append(done["worker_wall_seconds"])
                    continue
                def status(state):
                    value=dict(status=state,active_run=cell["id"],active_output=str(folder),
                        completed_runs=len(completed),total_runs=len(plan["jobs"]),updated_at=now(),
                        completed_ids=completed.copy(),pid=os.getpid())
                    if durations:
                        value["rough_remaining_hours"]=sum(durations)/len(durations)*(len(plan["jobs"])-len(completed))/3600
                        value["estimate_scope"]="Mean completed worker wall time; early estimates may be biased by family/case order; excludes final analysis"
                    if (folder/"progress.json").exists():value["active_progress"]=read(folder/"progress.json")
                    save_json(output/"progress.json",value)
                    return value
                if (output/"PAUSE_REQUEST.json").exists():status("paused");return
                pause=folder/"PAUSE_REQUEST.json"
                if pause.exists() and read(pause).get("validation_controller") is True:pause.unlink()
                expected=dict(mode="Run",output=cell["output"],config=cell["config"],dependency_path=job["dependency_path"])
                assert read(cell["worker_job"])==expected,"Worker routing changed"
                status("running")
                print(now(),"RUN",len(completed)+1,"/",len(plan["jobs"]),cell["id"],flush=True)
                python=str(Path(sys.executable).with_name("python.exe"))
                stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                bootstrap=output/"bootstrap"
                bootstrap.mkdir(exist_ok=True)
                with (bootstrap/(cell["id"]+"_"+stamp+".log")).open("a",encoding="utf-8") as stream:
                    kwargs=dict(cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    if os.name=="nt":kwargs["creationflags"]=subprocess.CREATE_NO_WINDOW
                    process=subprocess.Popen([python,str(Path(__file__).resolve()),"--job",cell["worker_job"]],**kwargs)
                    last=-1
                    while process.poll() is None:
                        if (output/"PAUSE_REQUEST.json").exists() and folder.exists():
                            save_json(pause,dict(validation_controller=True))
                        value=status("running")
                        count=value.get("active_progress",{}).get("requests",0)
                        if count!=last:
                            print(now(),"PROGRESS",cell["id"],count,"/ 10000",flush=True)
                            last=count
                        time.sleep(4)
                    if process.returncode:raise RuntimeError(f"{cell['id']} failed: exit {process.returncode}")
                if not (folder/"DONE.json").exists():status("paused");return
                done=read(folder/"DONE.json")
                completed.append(cell["id"])
                durations.append(done["worker_wall_seconds"])
                print(now(),"COMPLETE",len(completed),"/",len(plan["jobs"]),cell["id"],flush=True)
            save_json(output/"progress.json",dict(status="analyzing",completed_runs=180,total_runs=180,updated_at=now()))
            verify_sources()
            from windfarm_validation.analysis import run
            result=run(plan)
            save_json(output/"DONE.json",result)
            save_json(output/"progress.json",dict(status="complete",completed_runs=180,total_runs=180,updated_at=now()))
            print(now(),"CONFIRMATION COMPLETE; independent audits and figures finished",flush=True)
        except BaseException:
            error=dict(status="failed",error=traceback.format_exc(),completed_runs=len(completed),updated_at=now())
            stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_json(output/("FAILED_"+stamp+".json"),error)
            save_json(output/"progress.json",error)
            raise
        finally:
            if inhibited:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    args=parser.parse_args()
    job=read(args.job)
    root=Path(__file__).resolve().parent.parent
    os.chdir(root)
    sys.path.insert(0,str(root))
    if job.get("dependency_path"):sys.path.insert(0,job["dependency_path"])
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg",WINDFARM_VALIDATION_BACKGROUND="1")
    output=Path(job["output"])
    output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sys.stdin=open(os.devnull)
    sys.stdout=sys.stderr=(output/("worker_"+stamp+".log")).open("a",encoding="utf-8",buffering=1)
    print(datetime.datetime.now().astimezone().isoformat(),"WORKER START",os.getpid(),job["mode"],flush=True)
    try:
        if job["mode"]=="Check":
            from windfarm_research_v6.lifecycle import FileLock
            from windfarm_validation.checks import run
            with FileLock(output/"checks.lock"):run(output)
        elif job["mode"]=="Run":
            from windfarm_validation.runtime import run_one
            run_one(read(job["config"]),output)
        elif job["mode"]=="Compare":campaign(job)
        else:raise ValueError("Unknown mode")
    except BaseException:
        traceback.print_exc()
        raise


if __name__=="__main__":main()
