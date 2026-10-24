"""Three-method terrain study; reuse the verified offline lifecycle without editing it."""
from pathlib import Path
import os
import sys
import time
import hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
OUT=HERE/"results/terrain_large_campaign"
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
sys.path.insert(0,str(ROOT/"windfarm_terrain_v5"))
import bootstrap_v5
import repair_large_campaign as lifecycle
from repair_large_campaign import read,save,sha,dump_checkpoint
from pilot import lock_file,now


def config():
    cfg=read(HERE/"terrain_large_plan.json")
    assert cfg["variants"]==["CCwake_base","LM_base","DGEA_base"]
    assert cfg["total_runs"]==60 and cfg["evaluations"]==10000
    assert cfg["seeds"]==list(range(801,811))
    assert len(cfg["cases"])==2 and all(c["terrain"]=="ridge" and c["candidates"]==1600 and c["turbines"]==400 for c in cfg["cases"])
    cfg.update(workers=4,checkpoint_requests=250,output=str(OUT),runtime_status="frozen_for_launch")
    return cfg


def specs(cfg):
    return [dict(c,variant=v,seed=s,name=f"{c['scenario']}_{c['terrain']}_M{c['candidates']}_N{c['turbines']}_{v}_s{s}")
            for s in cfg["seeds"] for c in cfg["cases"] for v in cfg["variants"]]


def signature():
    pre=read(HERE/"results/terrain_large_preflight/DONE.json");assert pre["passed"]
    result=pre["source_hashes"].copy()
    for p,expected in result.items():assert sha(Path(p))==expected,p
    for p in [Path(__file__),HERE/"terrain_large_model.py",HERE/"terrain_large_analysis.py",HERE/"results/terrain_large_preflight/DONE.json"]:
        result[str(p)]=sha(p)
    return result


def worker(spec,cfg):
    import numpy as np
    import dill
    from shapely.geometry import Point
    from variants import Engine,VARIANTS,configure
    from terrain_large_model import RecordedTerrainFarm
    from algorithms import NDS
    from fast_audit import independent_check
    folder=OUT/spec["name"];folder.mkdir(exist_ok=True)
    with lock_file(folder/"run.lock"):
        if (folder/"DONE.json").exists():return
        assert signature()==read(OUT/"manifest.json")
        configure({});method=VARIANTS[spec["variant"]]["method"]
        prior=0.;restore_seconds=0.
        if (folder/"last.pkl").exists():
            started_restore=time.perf_counter()
            with (folder/"last.pkl").open("rb") as f:payload=dill.load(f)
            engine=payload["engine"];prior=payload["elapsed_seconds"]
            init_seconds=payload["initialization_seconds"]
            restore_seconds=payload["checkpoint_restore_seconds"]+time.perf_counter()-started_restore
            assert engine.method==method and engine.farm.terrain.kind=="ridge"
            print(now(),"RESUME",spec["name"],len(engine.farm.records),flush=True)
        else:
            assert not (folder/"progress.json").exists(),"Progress without checkpoint; refusing fresh restart"
            init_start=time.perf_counter()
            farm=RecordedTerrainFarm(spec["scenario"],1600,400,terrain_kind="ridge",resolution=50.,slope_limit_deg=15.)
            assert all(not farm.terrain.excluded.covers(Point(p)) for p in farm.xy)
            np.testing.assert_array_equal(farm.xy,np.load(ROOT/"windfarm_terrain_v5/data/ridge_50m/candidates_M1600.npy"))
            engine=Engine(farm,method,spec["seed"],10000,100)
            init_seconds=time.perf_counter()-init_start
            assert len(farm.records)==len(farm.initial_keys)==100
            dump_checkpoint(folder,engine,0.,init_seconds,restore_seconds)
            print(now(),"START",spec["name"],100,flush=True)
        started=time.perf_counter();last=-1
        while True:
            count=len(engine.farm.records)
            if count-last>=cfg["checkpoint_requests"] or engine.remaining==0:
                elapsed=prior+time.perf_counter()-started
                dump_checkpoint(folder,engine,elapsed,init_seconds,restore_seconds)
                save(folder/"progress.json",dict(requests=count,budget=10000,unique_layouts=len(engine.farm.cache),
                    pid=os.getpid(),elapsed_seconds=elapsed,initialization_seconds=init_seconds,updated_at=now()))
                print(now(),"PROGRESS",spec["name"],count,flush=True);last=count
                if (OUT/"PAUSE_REQUEST.json").exists():return
            if not engine.remaining:break
            before=len(engine.farm.records);engine.step();assert before<len(engine.farm.records)<=10000
        farm=engine.farm;records=farm.records
        selected=np.array([np.frombuffer(r[0],dtype="<u2") for r in records])
        metrics=np.array([r[1] for r in records]);cv=np.array([r[2] for r in records])
        unique={}
        for i,r in enumerate(records):
            if r[2]==0:unique.setdefault(r[0],i)
        ids=np.array(list(unique.values()),int);assert len(ids)
        front=ids[NDS.do(metrics[ids,:3]*[-1,1,1],only_non_dominated_front=True)]
        assert all(independent_check(farm.scene,farm.xy[selected[i]]) for i in ids)
        assert all(not farm.terrain.excluded.covers(Point(p)) for p in farm.xy)
        assert np.all(cv[:100]==0)
        np.savez_compressed(folder/"evaluations.npz",selected=selected,metrics=metrics,cv=cv,candidate_xy=farm.xy,
            pareto_request_indices=front,raw_conflict_pairs=[r[3] for r in records],repair_success=[r[4] for r in records],
            cache_hit=[r[5] for r in records],repair_replacements=[r[6] for r in records],key_diagnostics=np.array(farm.diagnostics),
            initial_keys=np.array(farm.initial_keys))
        save(folder/"trace.json",engine.trace);save(folder/"case_metadata.json",farm.case_metadata())
        summary=dict(spec,requests=len(records),unique_layouts=len(farm.cache),unique_feasible_layouts=len(ids),
            duplicate_rate=float(np.mean([r[5] for r in records])),raw_feasible_rate=float(np.mean([r[3]==0 for r in records])),
            post_feasible_rate=float(np.mean(cv==0)),mean_population=float(np.mean([r["population"] for r in engine.trace])),
            pareto_count=len(front),elapsed_seconds=prior+time.perf_counter()-started,initialization_seconds=init_seconds,
            checkpoint_restore_seconds=restore_seconds,independent_geometry_passed=True,independent_slope_mask_passed=True,
            initial_keys_sha256=hashlib.sha256(np.array(farm.initial_keys).tobytes()).hexdigest(),
            initial_layouts_sha256=hashlib.sha256(selected[:100].tobytes()).hexdigest(),
            initial_metrics_sha256=hashlib.sha256(metrics[:100].tobytes()).hexdigest(),finished_at=now())
        save(folder/"DONE.json",summary);print(now(),"DONE",spec["name"],flush=True)


if __name__=="__main__":
    # Bind the existing scheduled controller to this explicitly fixed 60-run
    # campaign. Only this process's module globals change; files stay immutable.
    import repair_large_analysis as dispatch
    from terrain_large_analysis import run
    dispatch.run=run
    lifecycle.OUT=OUT
    lifecycle.__file__=str(Path(__file__).resolve())
    lifecycle.validated_config=config
    lifecycle.signature=signature
    lifecycle.specs=specs
    lifecycle.worker=worker
    lifecycle.main()
