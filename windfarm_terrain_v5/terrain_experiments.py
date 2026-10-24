"""Separate, checkpointed terrain campaign reusing the verified V4 worker."""
import bootstrap_v5
from pathlib import Path
import argparse,json,hashlib,os,sys,time,subprocess,datetime,traceback
import pilot as P
from launch import save,stamp
ROOT=Path(__file__).resolve().parent;OUT=ROOT/"results/terrain_experiments"
CONFIG=ROOT/"terrain_config.json"
ORIGINAL_SIGNATURE=P.fingerprint

def signature():
    paths=[ROOT/f for f in ["terrain.py","geometry.py","model_terrain.py","terrain_experiments.py","terrain_analysis.py","terrain_config.json"]]
    paths+=[bootstrap_v5.V4/"posthoc/confirm_variants.py"]
    paths+=[ROOT/"results/terrain_audit/surface_quadrature_orders.json"]
    paths+=list((ROOT/"data").glob("*_50m/candidates_*.npy"))
    result=ORIGINAL_SIGNATURE()
    result.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result
def specs(cfg):
    return [dict(c,variant=v,seed=s,name=f"{c['scenario']}_{c['terrain']}_M{c['candidates']}_N{c['turbines']}_{v}_s{s}")
            for s in cfg["seeds"] for c in cfg["cases"] for v in cfg["variants"]]
def worker(spec,cfg):
    import numpy as np
    from shapely.geometry import Point
    from model_terrain import TerrainFarm
    sys.path.insert(0,str(bootstrap_v5.V4/"posthoc"))
    from confirm_variants import install
    install()
    import variants as V
    options=dict(V.VARIANTS[spec["variant"]],terrain_kind=spec["terrain"])
    V.VARIANTS[spec["variant"]]=options
    def factory(*args,**kwargs):
        farm=TerrainFarm(*args,**kwargs,terrain_kind=spec["terrain"],resolution=cfg["resolution_m"],slope_limit_deg=cfg["slope_limit_deg"])
        expected=np.load(ROOT/"data"/(spec["terrain"]+"_50m")/f"candidates_M{spec['candidates']}.npy")
        assert np.array_equal(farm.xy,expected),"Changed terrain candidate pool"
        assert all(not farm.terrain.excluded.covers(Point(p)) for p in farm.xy),"Independent slope polygon check failed"
        save(OUT/spec["name"]/"terrain_pool_check.json",dict(passed=True,candidates=len(farm.xy),terrain_sha256=farm.terrain.digest))
        return farm
    V.DiagnosticFarm=factory
    P.OUT=OUT;P.fingerprint=signature
    P.worker(spec,cfg)

def run(out):
    import numpy as np
    from model_terrain import TerrainFarm
    cfg=P.read(CONFIG);jobs=specs(cfg);assert len(jobs)==90
    assert P.read(ROOT/"results/terrain_audit/DONE.json")["status"]=="complete"
    assert P.read(ROOT/"results/model_audit/DONE.json")["status"]=="complete"
    assert P.read(ROOT/"results/pipeline_check/DONE.json")["status"]=="complete"
    current=signature()
    if (out/"manifest.json").exists():assert P.read(out/"manifest.json")==current,"Changed terrain sources"
    else:save(out/"manifest.json",current)
    save(out/"frozen_config.json",cfg)
    # Prepare shared immutable route matrices once, before the optimizer jobs.
    for i,c in enumerate(cfg["cases"]):
        save(out/"progress.json",dict(status="preparing_surface_routes",prepared_cases=i,total_cases=len(cfg["cases"]),
              current=c,completed=sum((out/s["name"]/"DONE.json").exists() for s in jobs),total=len(jobs),
              pid=os.getpid(),updated_at=stamp()))
        farm=TerrainFarm(c["scenario"],c["candidates"],c["turbines"],terrain_kind=c["terrain"],
                         resolution=cfg["resolution_m"],slope_limit_deg=cfg["slope_limit_deg"])
        farm.ensure_ground()
        metadata=farm.case_metadata()
        save(out/(f"prepared_{c['terrain']}_M{c['candidates']}.json"),metadata)
        print("PREPARED",c["terrain"],c["candidates"],flush=True)
    active={}
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",MPLBACKEND="Agg")
    while True:
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is not None:
                del active[name]
                if code!=0:raise RuntimeError(f"Terrain worker {name} failed; checkpoint retained")
        done=[s for s in jobs if (out/s["name"]/"DONE.json").exists()]
        if len(done)==len(jobs):break
        paused=(out/"PAUSE_REQUEST.json").exists()
        pending=[];external=[]
        for s in jobs:
            folder=out/s["name"];folder.mkdir(exist_ok=True)
            if (folder/"DONE.json").exists() or s["name"] in active:continue
            if P.busy(folder/"run.lock"):external.append(s["name"])
            else:pending.append(s)
        if not paused:
            for s in pending[:max(0,cfg["workers"]-len(active)-len(external))]:
                folder=out/s["name"]
                log=folder/("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
                with log.open("ab",buffering=0) as f:
                    proc=subprocess.Popen([sys.executable,"-u",str(__file__),"--run",s["name"]],cwd=ROOT,env=env,
                        stdin=subprocess.DEVNULL,stdout=f,stderr=f,close_fds=True,
                        creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
                active[s["name"]]=proc
                save(folder/"launch.json",dict(pid=proc.pid,log=str(log),started_at=stamp()))
                print("LAUNCH",s["name"],proc.pid,flush=True)
        save(out/"progress.json",dict(status="pausing" if paused else "running",completed=len(done),total=len(jobs),
             active=list(active)+external,pid=os.getpid(),updated_at=stamp()))
        if paused and not active and not external:
            save(out/"progress.json",dict(status="paused",completed=len(done),total=len(jobs),updated_at=stamp()))
            return
        time.sleep(2)
    save(out/"experiments_DONE.json",dict(runs=len(jobs),requests=len(jobs)*cfg["evaluations"],finished_at=stamp()))
    from terrain_analysis import run as analyze
    analyze(out,cfg)
    save(out/"DONE.json",dict(status="complete",runs=len(jobs),requests=len(jobs)*cfg["evaluations"],finished_at=stamp()))
    save(out/"progress.json",dict(status="complete",completed=len(jobs),total=len(jobs),updated_at=stamp()))
    print("TERRAIN EXPERIMENTS AND ANALYSIS COMPLETE",flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);args=ap.parse_args()
    cfg=P.read(CONFIG);spec=next(s for s in specs(cfg) if s["name"]==args.run)
    try:worker(spec,cfg)
    except Exception:
        folder=OUT/args.run;folder.mkdir(exist_ok=True)
        save(folder/("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),
             dict(traceback=traceback.format_exc(),failed_at=stamp()))
        raise
