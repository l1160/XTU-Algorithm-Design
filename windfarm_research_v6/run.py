"""Run exactly one specified comparison cell, with full-state resume."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import pickle
import sys
import time
import numpy as np

from .compat import ROOT, METHODS, legacy as A
from .engines import make_engine
from .model import OverlapFarm
from .lifecycle import FileLock, now, save_json, sha


def validate_config(value):
    allowed = {"method","scenario","terrain","candidates","turbines","seed","candidate_seed",
               "population","requests","checkpoint_requests","incremental","layout_diversity","compound","max_attempts"}
    if set(value)-allowed:
        raise ValueError(f"Unknown configuration keys: {set(value)-allowed}")
    cfg = dict(method="CCGDE3_problem",scenario="obs_00",terrain="flat",candidates=1600,
               turbines=400,seed=901,candidate_seed=20261012,population=100,requests=10000,
               checkpoint_requests=250,incremental=True,layout_diversity=True,compound=True,max_attempts=24)
    cfg.update(value)
    if cfg["method"] not in METHODS or cfg["terrain"] not in ("flat","rolling","ridge"):
        raise ValueError("Unknown method or terrain")
    available = {p.stem for p in (ROOT/"windflo_standard/data/Scenarios").glob("*.xml")}
    if cfg["scenario"] not in available:
        raise ValueError("Unknown input scenario")
    for name in ("candidates","turbines","seed","candidate_seed","population","requests","checkpoint_requests","max_attempts"):
        if type(cfg[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    for name in ("incremental","layout_diversity","compound"):
        if type(cfg[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if not (2 <= cfg["turbines"] < cfg["candidates"] < 65536):
        raise ValueError("Require 2<=N<M<65536")
    if cfg["population"] < 4 or cfg["requests"] < cfg["population"] or cfg["checkpoint_requests"] < 1 or cfg["max_attempts"] < 2:
        raise ValueError("Invalid population or budget")
    if cfg["seed"] < 0 or cfg["candidate_seed"] < 0:
        raise ValueError("Seeds must be nonnegative")
    if not cfg["method"].endswith("_problem") and (not cfg["layout_diversity"] or not cfg["compound"] or cfg["max_attempts"] != 24):
        raise ValueError("Problem-specific switches are not baseline switches")
    return cfg


def signature(cfg):
    paths = list((ROOT/"windfarm_research_v6").glob("*.py"))
    for directory in ("windfarm_study_v3","windfarm_research_v4","windfarm_terrain_v5","windflo_standard"):
        paths.extend((ROOT/directory).glob("*.py"))
    paths += [ROOT/"windfarm_study_v3/economics.json", ROOT/"windfarm_terrain_v5/results/terrain_audit/surface_quadrature_orders.json",
              ROOT/"windflo_standard/data/Scenarios"/(cfg["scenario"]+".xml"), ROOT/"requirements.txt"]
    return {"config":cfg,"sources":{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))},
            "python":list(sys.version_info[:3]),
            "packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","shapely","pymoo")}}


def save_checkpoint(folder,payload):
    temp = folder/("last.pkl."+str(os.getpid())+".tmp")
    with temp.open("wb") as stream:
        pickle.dump(payload,stream,protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp,folder/"last.pkl")


def run_one(config,output,stop_after_steps=None):
    cfg = validate_config(config)
    folder = Path(output).resolve()
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(folder/"run.lock"):
        current = signature(cfg)
        manifest = folder/"manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != current:
                raise ValueError("Configuration/source/environment changed; preserve this run and use a new output directory")
        else:
            if any((folder/name).exists() for name in ("last.pkl","progress.json","DONE.json","evaluations.npz")):
                raise ValueError("Existing results lack a matching manifest; refusing fresh start")
            save_json(manifest,current)
        if (folder/"DONE.json").exists():
            print(now(),"ALREADY COMPLETE",str(folder),flush=True)
            return json.loads((folder/"DONE.json").read_text(encoding="utf-8"))

        segment_wall,segment_cpu = time.perf_counter(),time.process_time()
        restored = 0.
        previous_wall = previous_cpu = 0.
        if (folder/"last.pkl").exists():
            restore_start = time.perf_counter()
            with (folder/"last.pkl").open("rb") as stream:
                state = pickle.load(stream)
            if state["signature"] != current:
                raise ValueError("Checkpoint signature does not match this run")
            engine = state["engine"]
            previous_wall,previous_cpu = state["worker_wall_seconds"],state["worker_cpu_seconds"]
            initialization = state["initialization_wall_seconds"]
            restored = state["restore_wall_seconds"]+time.perf_counter()-restore_start
            print(now(),"RESUME",cfg["method"],len(engine.farm.records),flush=True)
        else:
            if (folder/"progress.json").exists():
                raise ValueError("Progress exists without full last.pkl; refusing a fresh restart")
            farm = OverlapFarm(cfg["scenario"],cfg["candidates"],cfg["turbines"],cfg["candidate_seed"],
                               terrain_kind=cfg["terrain"],initial_size=cfg["population"],incremental=cfg["incremental"])
            options = {k:cfg[k] for k in ("layout_diversity","compound","max_attempts")} if cfg["method"].endswith("_problem") else {}
            engine = make_engine(farm,cfg["method"],cfg["seed"],cfg["requests"],cfg["population"],**options)
            initialization = time.perf_counter()-segment_wall
            print(now(),"START",cfg["method"],len(engine.farm.records),flush=True)
        # Save the fully evaluated initialization before the first optimizer step,
        # even when the checkpoint interval exceeds the population size.
        last_saved = len(engine.farm.records)-cfg["checkpoint_requests"]
        steps = 0

        def checkpoint(status):
            wall = previous_wall+time.perf_counter()-segment_wall
            cpu = previous_cpu+time.process_time()-segment_cpu
            payload = dict(engine=engine,signature=current,worker_wall_seconds=wall,
                           worker_cpu_seconds=cpu,initialization_wall_seconds=initialization,
                           restore_wall_seconds=restored)
            save_checkpoint(folder,payload)
            progress = dict(status=status,method=cfg["method"],requests=len(engine.farm.records),budget=cfg["requests"],
                            unique_layouts=len(engine.farm.cache),new_feasible_evaluations=engine.farm.statistics["new_feasible_evaluations"],
                            worker_wall_seconds=wall,worker_cpu_seconds=cpu,initialization_wall_seconds=initialization,
                            restore_wall_seconds=restored,wall_cpu_gap_seconds=max(0.,wall-cpu),pid=os.getpid(),updated_at=now(),
                            timing_scope="Wall time includes interruption and contention; CPU includes only this single-thread worker; do not infer dedicated-host speed from these fields alone")
            save_json(folder/"progress.json",progress)
            print(now(),"PROGRESS",cfg["method"],progress["requests"],"unique",progress["unique_layouts"],status,flush=True)
            return progress

        while True:
            count = len(engine.farm.records)
            stopping = ((folder/"PAUSE_REQUEST.json").exists() or
                        (stop_after_steps is not None and steps>=stop_after_steps))
            if count-last_saved>=cfg["checkpoint_requests"] or not engine.remaining or stopping:
                progress = checkpoint("paused" if stopping and engine.remaining else "running")
                last_saved = count
            if stopping and engine.remaining:
                return progress
            if not engine.remaining:
                break
            engine.step()
            steps += 1
            if not count < len(engine.farm.records) <= cfg["requests"]:
                raise AssertionError("The engine failed to make bounded budget progress")
        farm = engine.farm
        records = farm.records
        choices = np.array([np.frombuffer(r[0],dtype="<u2") for r in records])
        metrics = np.array([r[1] for r in records])
        cv = np.array([r[2] for r in records])
        unique = {}
        for i,r in enumerate(records):
            if r[2] == 0:
                unique.setdefault(r[0],i)
        ids = np.array(list(unique.values()),int)
        if not len(ids):
            raise ValueError("No feasible output; do not mark complete")
        front = ids[A.NDS.do(metrics[ids,:3]*[-1,1,1],only_non_dominated_front=True)]
        from fast_audit import independent_check
        if not all(independent_check(farm.scene,farm.xy[choices[i]]) and farm.terrain.allowed_mask(farm.xy[choices[i]]).all() for i in front):
            raise AssertionError("Independent final Pareto geometry check failed")
        temp = folder/"evaluations.npz.tmp"
        with temp.open("wb") as stream:
            np.savez_compressed(stream,selected=choices,metrics=metrics,cv=cv,candidate_xy=farm.xy,
                                pareto_request_indices=front,cache_hit=[r[5] for r in records],
                                raw_conflict_pairs=[r[3] for r in records])
        os.replace(temp,folder/"evaluations.npz")
        save_json(folder/"trace.json",engine.trace)
        save_json(folder/"case_metadata.json",farm.case_metadata())
        summary = dict(progress,status="complete",method=cfg["method"],config=cfg,
                       statistics=dict(farm.statistics),operator_counts=dict(getattr(engine,"operator_counts",{})),
                       component_wall_seconds=dict(farm.components),pareto_count=len(front),unique_feasible_layouts=len(ids),
                       duplicate_request_rate=float(np.mean([r[5] for r in records])),post_feasible_rate=float(np.mean(cv==0)),
                       initial_keys_sha256=__import__("hashlib").sha256(np.array(farm.initial_keys).astype("<f8").tobytes()).hexdigest(),
                       initial_layouts_sha256=__import__("hashlib").sha256(choices[:cfg["population"]].astype("<u2").tobytes()).hexdigest(),
                       independent_pareto_geometry_passed=True,finished_at=now(),
                       result_scope="One run on the V6 proxy model; no superiority or convergence claim")
        save_json(folder/"DONE.json",summary)
        save_json(folder/"progress.json",dict(progress,status="complete",updated_at=now()))
        print(now(),"DONE",cfg["method"],len(records),flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--check-config",action="store_true")
    args = parser.parse_args()
    cfg = validate_config(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
    if args.check_config:
        print(json.dumps({"config":cfg,"output":str(Path(args.output).resolve()),"optimization_started":False},ensure_ascii=False))
        return
    if os.environ.get("WINDFARM_V6_BACKGROUND") != "1":
        parser.error("Use Start-V6.ps1 for detached execution; --check-config is read-only")
    run_one(cfg,args.output)


if __name__ == "__main__":
    main()
