"""Detached six-process mechanism screening with complete per-run checkpoints."""
import bootstrap_v4
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback,msvcrt
HERE=Path(__file__).resolve().parent;OUT=HERE/"results/pilot"
read=lambda p:json.loads(p.read_text(encoding="utf-8"))
def now():return datetime.datetime.now().astimezone().isoformat()
def save(p,obj):
    t=p.with_suffix(p.suffix+".tmp")
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");t.replace(p)
def fingerprint():
    paths=list(HERE.glob("*.py"))+[HERE/"pilot_config.json"]
    paths += [bootstrap_v4.V3/f for f in ["algorithms.py","model.py","physical.py","economics.json"]]
    paths += [bootstrap_v4.V3.parent/"windflo_standard/evaluator.py",bootstrap_v4.V3.parent/"windflo_standard/data/Scenarios/obs_00.xml"]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
def lock_file(p):
    f=p.open("a+b");f.seek(0)
    if not f.read(1):f.write(b"0");f.flush()
    f.seek(0)
    try:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:f.close();raise
    return f
def busy(p):
    try:f=lock_file(p);f.close();return False
    except OSError:return True
def specs(cfg):
    return [dict(case,variant=v,seed=s,name=f"{case['scenario']}_M{case['candidates']}_N{case['turbines']}_{v}_s{s}")
            for s in cfg["seeds"] for case in cfg["cases"] for v in cfg["variants"]]
def worker(spec,cfg):
    import numpy as np
    import dill
    from variants import VARIANTS,DiagnosticFarm,Engine,configure
    from algorithms import NDS
    from fast_audit import independent_check
    d=OUT/spec["name"];d.mkdir(exist_ok=True)
    with lock_file(d/"run.lock"):
        if (d/"DONE.json").exists():return
        assert fingerprint()==read(OUT/"manifest.json"),"Sources changed"
        options=VARIANTS[spec["variant"]];configure(options)
        prior=0.
        if (d/"last.pkl").exists():
            with (d/"last.pkl").open("rb") as f:payload=dill.load(f)
            engine=payload["engine"];prior=payload["elapsed_seconds"]
            print("RESUME",spec["name"],len(engine.farm.records),flush=True)
        else:
            farm=DiagnosticFarm(spec["scenario"],spec["candidates"],spec["turbines"],options=options)
            engine=Engine(farm,options["method"],spec["seed"],cfg["evaluations"],cfg["population"])
            print("START",spec["name"],len(engine.farm.records),flush=True)
        started=time.perf_counter();last=-1
        while True:
            count=len(engine.farm.records)
            if count==cfg["evaluations"] or count-last>=cfg["checkpoint_requests"]:
                elapsed=prior+time.perf_counter()-started
                p=d/"last.pkl";temp=d/"last.pkl.tmp"
                with temp.open("wb") as f:dill.dump(dict(engine=engine,elapsed_seconds=elapsed),f)
                temp.replace(p)
                save(d/"progress.json",dict(requests=count,budget=cfg["evaluations"],pid=os.getpid(),elapsed_seconds=elapsed,updated_at=now()))
                print("PROGRESS",spec["name"],count,"unique",len(engine.farm.cache),flush=True)
                last=count
                if (OUT/"PAUSE_REQUEST.json").exists():print("PAUSED",flush=True);return
            if engine.remaining==0:break
            engine.step()
        farm=engine.farm;records=farm.records
        selected=np.array([np.frombuffer(r[0],dtype="<u2") for r in records])
        metrics=np.array([r[1] for r in records]);cv=np.array([r[2] for r in records])
        unique={}
        for i,r in enumerate(records):
            if r[2]==0:unique.setdefault(r[0],i)
        ids=np.array(list(unique.values()),int)
        front=ids[NDS.do(metrics[ids,:3]*[-1,1,1],only_non_dominated_front=True)]
        assert all(independent_check(farm.scene,farm.xy[selected[i]]) for i in ids)
        np.savez_compressed(d/"evaluations.npz",selected=selected,metrics=metrics,cv=cv,candidate_xy=farm.xy,
                            pareto_request_indices=front,raw_conflict_pairs=[r[3] for r in records],
                            repair_success=[r[4] for r in records],cache_hit=[r[5] for r in records],
                            repair_replacements=[r[6] for r in records],key_diagnostics=np.array(farm.diagnostics))
        save(d/"trace.json",engine.trace)
        save(d/"case_metadata.json",farm.case_metadata())
        summary=dict(spec,requests=len(records),unique_layouts=len(farm.cache),unique_feasible_layouts=len(ids),
                     duplicate_rate=float(np.mean([r[5] for r in records])),raw_feasible_rate=float(np.mean([r[3]==0 for r in records])),
                     post_feasible_rate=float(np.mean(cv==0)),mean_replacements=float(np.mean([r[6] for r in records])),
                     mean_repair_key_fraction=float(np.mean(np.array(farm.diagnostics)[:,0])),
                     mean_population=float(np.mean([r["population"] for r in engine.trace if "population" in r])),
                     mean_velocity_nonzero=float(np.mean([r["nonzero_velocity_fraction"] for r in engine.trace if "nonzero_velocity_fraction" in r])),
                     pareto_count=len(front),elapsed_seconds=prior+time.perf_counter()-started,independent_geometry_passed=True,
                     initial_layouts_sha256=hashlib.sha256(selected[:100].tobytes()).hexdigest(),finished_at=now())
        save(d/"DONE.json",summary);print("DONE",spec["name"],flush=True)
def controller(cfg):
    with lock_file(OUT/"worker.lock"):
        signature=fingerprint()
        if (OUT/"manifest.json").exists():
            assert read(OUT/"manifest.json")==signature,"Sources changed; preserve existing experiment"
        else:save(OUT/"manifest.json",signature)
        save(OUT/"frozen_config.json",cfg)
        if not (OUT/"preflight_DONE.json").exists():
            from preflight import run
            report=run();save(OUT/"preflight_DONE.json",report)
        jobs=specs(cfg);assert len(jobs)==72
        active={};handles={};all_started=False
        while True:
            done=[s for s in jobs if (OUT/s["name"]/"DONE.json").exists()]
            for name,p in list(active.items()):
                code=p.poll()
                if code is not None:
                    handles.pop(name).close();active.pop(name)
                    if code!=0:raise RuntimeError(f"Worker {name} failed with {code}; its logs/checkpoint preserved")
            if len(done)==len(jobs):break
            paused=(OUT/"PAUSE_REQUEST.json").exists()
            external=[]
            pending=[]
            for s in jobs:
                d=OUT/s["name"]
                if (d/"DONE.json").exists() or s["name"] in active:continue
                d.mkdir(exist_ok=True)
                if busy(d/"run.lock"):external.append(s["name"])
                else:pending.append(s)
            if not paused:
                for s in pending[:max(0,cfg["workers"]-len(active)-len(external))]:
                    d=OUT/s["name"];stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    log=d/("worker_"+stamp+".log");f=log.open("ab",buffering=0)
                    p=subprocess.Popen([sys.executable,"-u",str(__file__),"--run",s["name"]],cwd=HERE,
                                       stdin=subprocess.DEVNULL,stdout=f,stderr=f,
                                       creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
                    active[s["name"]]=p;handles[s["name"]]=f
                    save(d/"launch.json",dict(pid=p.pid,log=str(log),started_at=now()))
                    print("LAUNCH",s["name"],p.pid,flush=True)
            progress=dict(status="running" if not paused else "pausing",completed=len(done),total=len(jobs),
                          active=list(active)+external,pid=os.getpid(),updated_at=now())
            save(OUT/"progress.json",progress)
            if paused and not active and not external:save(OUT/"progress.json",dict(progress,status="paused"));return
            time.sleep(2)
        save(OUT/"experiments_DONE.json",dict(runs=len(jobs),requests=len(jobs)*cfg["evaluations"],finished_at=now()))
        from analyze_pilot import run
        run()
        save(OUT/"DONE.json",dict(status="complete",runs=len(jobs),requests=len(jobs)*cfg["evaluations"],stage="mechanism_screening",finished_at=now()))
        save(OUT/"progress.json",dict(status="complete",completed=len(jobs),total=len(jobs),updated_at=now()))
        print("PILOT AND ANALYSIS COMPLETE",flush=True)
if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument("--run");args=p.parse_args();cfg=read(HERE/"pilot_config.json")
    try:
        if args.run:worker(next(s for s in specs(cfg) if s["name"]==args.run),cfg)
        else:controller(cfg)
    except Exception:
        d=OUT/args.run if args.run else OUT
        save(d/("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),dict(traceback=traceback.format_exc(),failed_at=now()))
        raise
