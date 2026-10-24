"""Prespecified 60-run LMOCSO/DGEA spacing-repair study, scheduled offline."""
from pathlib import Path
import os
import sys
import argparse
import datetime
import hashlib
import json
import subprocess
import time
import traceback
import ctypes
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/repair_large_campaign"
CONFIG = HERE / "repair_large_config.json"
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
from pilot import lock_file, busy, specs, now


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    for attempt in range(30):
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(.1)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def signature():
    preflight = read(HERE / "results/repair_large_preflight/DONE.json")
    assert preflight["passed"] is True
    hashes = preflight["source_hashes"].copy()
    for path, expected in hashes.items():
        assert sha(Path(path)) == expected, f"Preflight source changed: {path}"
    for path in (Path(__file__), CONFIG, HERE / "repair_large_analysis.py",
                 HERE / "results/repair_large_preflight/DONE.json"):
        hashes[str(path)] = sha(path)
    return hashes


def validated_config():
    cfg = read(CONFIG)
    plan = read(HERE / cfg["protocol"])
    assert cfg["cases"] == [plan["case"]]
    assert cfg["seeds"] == plan["seeds"]
    assert cfg["population"] == plan["population"] == 100
    assert cfg["evaluations"] == plan["requests_per_run"] == 10000
    assert len(specs(cfg)) == plan["total_runs"] == 60
    assert {v["method"] for v in cfg["variant_definitions"].values()} == set(plan["algorithms"])
    assert len(cfg["paired_contrasts"]) == 4
    for pair, declared in zip(cfg["paired_contrasts"], plan["primary_contrasts"]):
        treatment, control = [cfg["variant_definitions"][v] for v in pair]
        assert treatment["method"] == control["method"] == declared["algorithm"]
        assert treatment["repair_mode"] == declared["treatment"] and control["repair_mode"] == declared["control"]
    return cfg


def dump_checkpoint(folder, engine, elapsed, init_seconds, restore_seconds):
    import dill
    tmp = folder / "last.pkl.tmp"
    with tmp.open("wb") as f:
        dill.dump(dict(engine=engine, elapsed_seconds=elapsed, initialization_seconds=init_seconds,
                       checkpoint_restore_seconds=restore_seconds), f)
    tmp.replace(folder / "last.pkl")


def worker(spec, cfg):
    import numpy as np
    import dill
    from variants import Engine
    from algorithms import NDS
    from fast_audit import independent_check
    from repair_ablation_large_model import RepairFarm, configure
    folder = OUT / spec["name"]
    folder.mkdir(exist_ok=True)
    with lock_file(folder / "run.lock"):
        if (folder / "DONE.json").exists():
            return
        assert signature() == read(OUT / "manifest.json")
        configure()
        definition = cfg["variant_definitions"][spec["variant"]]
        prior = 0.
        restore_seconds = 0.
        if (folder / "last.pkl").exists():
            restore_start = time.perf_counter()
            with (folder / "last.pkl").open("rb") as f:
                payload = dill.load(f)
            engine = payload["engine"]
            prior = payload["elapsed_seconds"]
            init_seconds = payload["initialization_seconds"]
            restore_seconds = payload["checkpoint_restore_seconds"] + time.perf_counter()-restore_start
            assert engine.method == definition["method"] and engine.farm.repair_mode == definition["repair_mode"]
            print(now(), "RESUME", spec["name"], len(engine.farm.records), flush=True)
        else:
            assert not (folder / "progress.json").exists(), "Progress exists without checkpoint; refusing fresh restart"
            init_start = time.perf_counter()
            farm = RepairFarm(spec["scenario"], spec["candidates"], spec["turbines"], repair_mode=definition["repair_mode"])
            engine = Engine(farm, definition["method"], spec["seed"], cfg["evaluations"], cfg["population"])
            init_seconds = time.perf_counter()-init_start
            assert len(farm.records) == 100
            dump_checkpoint(folder, engine, 0., init_seconds, restore_seconds)
            print(now(), "START", spec["name"], 100, flush=True)
        started = time.perf_counter()
        last_checkpoint = -1
        while True:
            count = len(engine.farm.records)
            if count-last_checkpoint >= cfg["checkpoint_requests"] or engine.remaining == 0:
                elapsed = prior + time.perf_counter()-started
                dump_checkpoint(folder, engine, elapsed, init_seconds, restore_seconds)
                save(folder / "progress.json", dict(requests=count, budget=cfg["evaluations"],
                    unique_layouts=len(engine.farm.cache), pid=os.getpid(), elapsed_seconds=elapsed,
                    initialization_seconds=init_seconds, updated_at=now()))
                print(now(), "PROGRESS", spec["name"], count, "unique", len(engine.farm.cache), flush=True)
                last_checkpoint = count
                if (OUT / "PAUSE_REQUEST.json").exists():
                    print("PAUSED after full checkpoint", flush=True)
                    return
            if not engine.remaining:
                break
            before = len(engine.farm.records)
            engine.step()
            assert before < len(engine.farm.records) <= cfg["evaluations"]
        farm = engine.farm
        records, details = farm.records, farm.repair_details
        assert len(records) == len(details) == 10000
        selected = np.array([np.frombuffer(r[0], dtype="<u2") for r in records])
        metrics = np.array([r[1] for r in records])
        cv = np.array([r[2] for r in records])
        unique = {}
        for i, r in enumerate(records):
            if r[2] == 0:
                unique.setdefault(r[0], i)
        ids = np.array(list(unique.values()), int)
        assert len(ids)
        front = ids[NDS.do(metrics[ids, :3]*[-1,1,1], only_non_dominated_front=True)]
        assert all(independent_check(farm.scene, farm.xy[selected[i]]) for i in ids)
        assert np.all(cv[:100] == 0)
        assert all(d["shared_initialization"] == (i < 100) for i, d in enumerate(details))
        assert all(d["mode"] == definition["repair_mode"] for d in details[100:])
        np.savez_compressed(folder / "evaluations.npz", selected=selected, metrics=metrics, cv=cv,
            candidate_xy=farm.xy, pareto_request_indices=front,
            raw_conflict_pairs=[r[3] for r in records], repair_success=[r[4] for r in records],
            cache_hit=[r[5] for r in records], repair_replacements=[r[6] for r in records],
            key_diagnostics=np.array(farm.diagnostics), initial_keys=np.array(farm.initial_keys),
            decode_seconds=[d["decode_seconds"] for d in details],
            evaluation_seconds=[d["evaluation_seconds"] for d in details],
            physical_evaluated=[d["physical_evaluated"] for d in details],
            greedy_fill_size=[-1 if d["greedy_fill_size"] is None else d["greedy_fill_size"] for d in details],
            greedy_invoked=[d["greedy_invoked"] for d in details],
            backtrack_attempts=[d["backtrack_attempts"] for d in details],
            backtrack_success=[d["backtrack_success"] for d in details],
            velocity_coordinates_reset=[d["velocity_coordinates_reset"] for d in details])
        save(folder / "trace.json", engine.trace)
        save(folder / "case_metadata.json", dict(farm.case_metadata(), repair_mode=definition["repair_mode"],
             shared_initial_requests=100, policy="Fixed key-order spacing repair; no forbidden-mask ablation"))
        post, dr = records[100:], details[100:]
        summary = dict(spec, method=definition["method"], repair_mode=definition["repair_mode"],
            requests=len(records), unique_layouts=len(farm.cache), unique_feasible_layouts=len(ids),
            duplicate_rate=float(np.mean([r[5] for r in records])),
            raw_feasible_rate=float(np.mean([r[3] == 0 for r in records])), post_feasible_rate=float(np.mean(cv == 0)),
            offspring_raw_feasible_rate=float(np.mean([r[3] == 0 for r in post])),
            offspring_post_feasible_rate=float(np.mean(cv[100:] == 0)),
            offspring_feasible_duplicates=sum(bool(r[5]) and r[2] == 0 for r in post),
            offspring_infeasible_duplicates=sum(bool(r[5]) and r[2] > 0 for r in post),
            offspring_greedy_calls=sum(d["greedy_invoked"] for d in dr),
            offspring_greedy_failures=sum(d["greedy_invoked"] and d["greedy_fill_size"] < farm.n for d in dr),
            offspring_backtrack_calls=sum(d["backtrack_attempts"] > 0 for d in dr),
            offspring_backtrack_attempts=sum(d["backtrack_attempts"] for d in dr),
            offspring_backtrack_successes=sum(d["backtrack_success"] for d in dr),
            offspring_decode_seconds=float(sum(d["decode_seconds"] for d in dr)),
            offspring_physical_evaluations=sum(d["physical_evaluated"] for d in dr),
            offspring_velocity_coordinates_reset=sum(d["velocity_coordinates_reset"] for d in dr),
            mean_population=float(np.mean([t["population"] for t in engine.trace])),
            pareto_count=len(front), elapsed_seconds=prior+time.perf_counter()-started,
            initialization_seconds=init_seconds, checkpoint_restore_seconds=restore_seconds,
            independent_geometry_passed=True,
            initial_keys_sha256=hashlib.sha256(np.array(farm.initial_keys).tobytes()).hexdigest(),
            initial_layouts_sha256=hashlib.sha256(selected[:100].tobytes()).hexdigest(),
            initial_metrics_sha256=hashlib.sha256(metrics[:100].tobytes()).hexdigest(), finished_at=now())
        save(folder / "DONE.json", summary)
        print(now(), "DONE", spec["name"], flush=True)


def controller(cfg):
    with lock_file(OUT / "worker.lock"):
        current = signature()
        if (OUT / "manifest.json").exists():
            assert read(OUT / "manifest.json") == current, "Source drift; no restart"
        else:
            save(OUT / "manifest.json", current)
            save(OUT / "frozen_config.json", cfg)
        jobs = specs(cfg)
        active = {}
        # Prevent idle sleep while this scheduled controller is active; never
        # inhibit shutdown, and release the request when the controller exits.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            while True:
                for name, process in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        del active[name]
                        if code != 0:
                            raise RuntimeError(f"Worker failed: {name}, exit {code}; checkpoint/logs retained")
                done = [s for s in jobs if (OUT / s["name"] / "DONE.json").exists()]
                if len(done) == len(jobs) and not active:
                    break
                pending, external = [], []
                for spec in jobs:
                    folder = OUT / spec["name"]
                    folder.mkdir(exist_ok=True)
                    if (folder / "DONE.json").exists() or spec["name"] in active:
                        continue
                    if busy(folder / "run.lock"):
                        external.append(spec["name"])
                    else:
                        pending.append(spec)
                paused = (OUT / "PAUSE_REQUEST.json").exists()
                if not paused:
                    for spec in pending[:max(0, cfg["workers"]-len(active)-len(external))]:
                        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--run", spec["name"]],
                            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                        active[spec["name"]] = process
                        print(now(), "LAUNCH", spec["name"], process.pid, flush=True)
                save(OUT / "progress.json", dict(status="pausing" if paused else "running", completed=len(done), total=60,
                    active=list(active)+external, pid=os.getpid(), updated_at=now()))
                if paused and not active and not external:
                    save(OUT / "progress.json", dict(status="paused", completed=len(done), total=60, updated_at=now()))
                    return
                time.sleep(3)
            save(OUT / "experiments_DONE.json", dict(runs=60, requests=600000, finished_at=now()))
            from repair_large_analysis import run
            run(OUT, cfg)
            assert signature() == current
            save(OUT / "DONE.json", dict(status="complete", runs=60, requests=600000, finished_at=now()))
            save(OUT / "progress.json", dict(status="complete", completed=60, total=60, updated_at=now()))
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run")
    args = parser.parse_args()
    cfg = validated_config()
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT / args.run if args.run else OUT
    folder.mkdir(exist_ok=True)
    logpath = folder / ("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    log = logpath.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    sys.stdin = open(os.devnull)
    try:
        save(folder / "launch.json", dict(pid=os.getpid(), started_at=now(), log=str(logpath)))
        print(now(), "WORKER" if args.run else "CONTROLLER", args.run or "", flush=True)
        if args.run:
            worker(next(s for s in specs(cfg) if s["name"] == args.run), cfg)
        elif not (OUT / "DONE.json").exists():
            controller(cfg)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        save(folder / ("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"), dict(error=error, failed_at=now()))
        raise
    finally:
        log.flush()


if __name__ == "__main__":
    main()
