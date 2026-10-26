"""Detached checks or the single four-run integrated comparison."""
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


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def campaign(job):
    from windfarm_research_v6.lifecycle import FileLock,save_json,sha,now
    from windfarm_research_v7.runner import signature
    root=Path(__file__).resolve().parent.parent
    plan=read(job["plan"]);output=Path(plan["output"])
    expected=["LMOCSO","LMOCSO_problem","DGEA","DGEA_problem"]
    assert [r["method"] for r in plan["jobs"]]==expected
    for path,value in plan["source_sha256"].items():assert sha(root/path)==value,path
    assert sha(plan["frozen_normalization"])==plan["normalization_sha256"]
    with FileLock(output/"campaign.lock"):
        if (output/"DONE.json").exists():return
        completed=[]
        inhibited=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=="nt" else False
        try:
            for cell in plan["jobs"]:
                config=read(cell["config"])
                assert sha(cell["config"])==cell["config_sha256"]
                folder=Path(cell["output"])
                if (folder/"DONE.json").exists():
                    assert read(folder/"manifest.json")==signature(config)
                    completed.append(cell["method"]);continue
                def status(state):
                    v=dict(status=state,active_method=cell["method"],completed_methods=completed.copy(),total_runs=4,updated_at=now())
                    if (folder/"progress.json").exists():v["active_progress"]=read(folder/"progress.json")
                    save_json(output/"progress.json",v)
                if (output/"PAUSE_REQUEST.json").exists():status("paused");return
                pause=folder/"PAUSE_REQUEST.json"
                if pause.exists() and read(pause).get("v7_controller") is True:
                    pause.unlink()
                routed=read(cell["worker_job"])
                if routed!={"mode":"Run","output":cell["output"],"config":cell["config"],"dependency_path":job["dependency_path"]}:
                    raise ValueError("Worker routing differs from the frozen comparison cell")
                status("running");print(now(),"RUN",cell["method"],flush=True)
                python=str(Path(sys.executable).with_name("python.exe"))
                stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                with (output/(cell["method"]+"_bootstrap_"+stamp+".log")).open("a",encoding="utf-8") as stream:
                    kwargs=dict(cwd=root,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                    if os.name=="nt":kwargs["creationflags"]=subprocess.CREATE_NO_WINDOW
                    process=subprocess.Popen([python,str(Path(__file__).resolve()),"--job",cell["worker_job"]],**kwargs)
                    while process.poll() is None:
                        if (output/"PAUSE_REQUEST.json").exists() and folder.exists():
                            save_json(folder/"PAUSE_REQUEST.json",{"v7_controller":True})
                        status("running");time.sleep(3)
                    if process.returncode:raise RuntimeError(f"{cell['method']} failed: exit {process.returncode}")
                if not (folder/"DONE.json").exists():status("paused");return
                completed.append(cell["method"]);print(now(),"COMPLETE",cell["method"],flush=True)
            save_json(output/"progress.json",dict(status="analyzing",completed_methods=completed,total_runs=4,updated_at=now()))
            from windfarm_research_v7.analysis import run
            result=run(plan);save_json(output/"DONE.json",result)
            save_json(output/"progress.json",dict(status="complete",completed_methods=completed,total_runs=4,updated_at=now()))
            print(now(),"COMPARISON COMPLETE",flush=True)
        except BaseException:
            save_json(output/"FAILED.json",dict(status="failed",error=traceback.format_exc(),updated_at=now()))
            raise
        finally:
            if inhibited:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--job",required=True)
    args=parser.parse_args();job=read(args.job)
    root=Path(__file__).resolve().parent.parent;os.chdir(root);sys.path.insert(0,str(root))
    if job.get("dependency_path"):sys.path.insert(0,job["dependency_path"])
    os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg",WINDFARM_V7_BACKGROUND="1")
    output=Path(job["output"]);output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sys.stdin=open(os.devnull)
    sys.stdout=sys.stderr=(output/("worker_"+stamp+".log")).open("a",encoding="utf-8",buffering=1)
    print(datetime.datetime.now().astimezone().isoformat(),"WORKER START",os.getpid(),job["mode"],flush=True)
    try:
        if job["mode"]=="Check":
            from windfarm_research_v7.checks import run
            run(output)
        elif job["mode"]=="Run":
            from windfarm_research_v7.runner import run_one
            run_one(read(job["config"]),output)
        elif job["mode"]=="Compare":campaign(job)
        else:raise ValueError("Unknown job mode")
    except BaseException:traceback.print_exc();raise


if __name__=="__main__":main()
