"""Independent-seed confirmation, using the frozen screening worker unchanged.

No launch occurs without an explicit config. All experiments run detached.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;V4=HERE.parent
sys.path.insert(0,str(V4))
import bootstrap_v4
import pilot as P

def signature(config,cfg):
    result=PILOT_SIGNATURE()
    paths=[Path(__file__),HERE/"confirm_analysis.py",HERE/"confirm_variants.py",config]
    paths += [bootstrap_v4.V3.parent/"windflo_standard/data/Scenarios"/(c["scenario"]+".xml") for c in cfg["cases"]]
    for p in paths:result[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result
PILOT_SIGNATURE=P.fingerprint

def launch_process(arguments,folder):
    log=folder/("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1",MPLBACKEND="Agg")
    with log.open("ab",buffering=0) as f:
        proc=subprocess.Popen([sys.executable,"-u",str(__file__),*arguments],cwd=HERE,env=env,
            stdin=subprocess.DEVNULL,stdout=f,stderr=f,close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    info=dict(pid=proc.pid,log=str(log),started_at=P.now())
    P.save(folder/"launch.json",info)
    return proc,info

def controller(config,cfg,out):
    jobs=P.specs(cfg)
    with P.lock_file(out/"worker.lock"):
        current=P.fingerprint()
        if (out/"manifest.json").exists():
            assert P.read(out/"manifest.json")==current,"Changed sources; preserve previous checkpoints"
        else:P.save(out/"manifest.json",current)
        P.save(out/"frozen_config.json",cfg)
        assert P.read(V4/"results/pilot/preflight_DONE.json")["passed"] is True
        assert P.read(V4/"results/confirm_preflight/DONE.json")["passed"] is True
        active={}
        while True:
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    del active[name]
                    if code!=0:raise RuntimeError(f"Failed {name}; checkpoint and logs preserved")
            done=[s for s in jobs if (out/s["name"]/"DONE.json").exists()]
            if len(done)==len(jobs):break
            paused=(out/"PAUSE_REQUEST.json").exists()
            external=[];pending=[]
            for s in jobs:
                folder=out/s["name"];folder.mkdir(exist_ok=True)
                if (folder/"DONE.json").exists() or s["name"] in active:continue
                if P.busy(folder/"run.lock"):external.append(s["name"])
                else:pending.append(s)
            if not paused:
                slots=max(0,cfg["workers"]-len(active)-len(external))
                for s in pending[:slots]:
                    proc,info=launch_process(["--config",str(config),"--run",s["name"]],out/s["name"])
                    active[s["name"]]=proc
                    print("LAUNCH",s["name"],info["pid"],flush=True)
            P.save(out/"progress.json",dict(status="pausing" if paused else "running",
                completed=len(done),total=len(jobs),active=list(active)+external,
                pid=os.getpid(),updated_at=P.now()))
            if paused and not active and not external:
                P.save(out/"progress.json",dict(status="paused",completed=len(done),total=len(jobs),updated_at=P.now()))
                return
            time.sleep(2)
        P.save(out/"experiments_DONE.json",dict(runs=len(jobs),requests=len(jobs)*cfg["evaluations"],finished_at=P.now()))
        from confirm_analysis import run
        run(out,cfg)
        P.save(out/"DONE.json",dict(status="complete",runs=len(jobs),finished_at=P.now()))
        P.save(out/"progress.json",dict(status="complete",completed=len(jobs),total=len(jobs),updated_at=P.now()))
        print("CONFIRMATION AND ANALYSIS COMPLETE",flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True);ap.add_argument("--worker",action="store_true")
    ap.add_argument("--run");ap.add_argument("--check-config",action="store_true")
    args=ap.parse_args();config=Path(args.config).resolve();cfg=P.read(config)
    from confirm_variants import install
    install()
    from variants import VARIANTS
    assert cfg["population"]==100 and cfg["evaluations"]>=2000 and cfg["evaluations"]%500==0
    assert 1<=cfg["workers"]<=6
    assert len(set(cfg["seeds"]))==len(cfg["seeds"])
    assert 6<=len(cfg["seeds"])<=16
    assert not set(cfg["seeds"])&{201,202,203,401,402,403}
    assert all(v in VARIANTS for v in cfg["variants"])
    assert len(set(cfg["variants"]))==len(cfg["variants"])
    assert all(t in cfg["variants"] and c in cfg["variants"] for t,c in cfg["paired_contrasts"])
    out=Path(cfg["output"]).resolve()
    assert out.parent==V4/"results" and out.name!="pilot","Separate results directory required"
    jobs=P.specs(cfg);assert len({s["name"] for s in jobs})==len(jobs)
    if args.check_config:
        print(json.dumps(dict(valid=True,runs=len(jobs),requests=len(jobs)*cfg["evaluations"],output=str(out))))
        return
    out.mkdir(exist_ok=True)
    P.OUT=out
    P.fingerprint=lambda:signature(config,cfg)
    try:
        if args.run:
            P.worker(next(s for s in jobs if s["name"]==args.run),cfg)
        elif args.worker:controller(config,cfg,out)
        elif (out/"DONE.json").exists():print("already_complete")
        elif P.busy(out/"worker.lock"):print("already_running")
        else:
            _,info=launch_process(["--config",str(config),"--worker"],out);print(json.dumps(info))
    except Exception:
        folder=out/args.run if args.run else out;folder.mkdir(exist_ok=True)
        P.save(folder/("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),
               dict(traceback=traceback.format_exc(),failed_at=P.now()))
        raise
if __name__=="__main__":main()
