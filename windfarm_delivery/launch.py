"""Detached postprocessing only. No optimization experiments are launched."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOBS = {
    "pilot_figures": ROOT / "windfarm_research_v4/results/pilot",
    "confirmation_figures": ROOT / "windfarm_research_v4/results/confirmation",
    "terrain_figures": ROOT / "windfarm_terrain_v5/results/terrain_experiments",
    "terrain_controls": ROOT / "windfarm_terrain_v5/results/terrain_experiments",
    "objective_witnesses": ROOT / "windfarm_research_v4/results/pilot",
    "cc_contract_audit": ROOT / "windfarm_research_v4/results/pilot",
    "encoding_replay": ROOT / "windfarm_research_v4/results/pilot",
    "scene_coverage": ROOT / "windfarm_research_v4/results/pilot",
    "replay_analysis": HERE / "results/encoding_replay",
    "confirmation_budget_diagnostics": ROOT / "windfarm_research_v4/results/confirmation",
    "terrain_descriptive": ROOT / "windfarm_terrain_v5/results/terrain_experiments",
    "terrain_layouts_verified": ROOT / "windfarm_terrain_v5/results/terrain_experiments",
    "confirmation_metric_review": ROOT / "windfarm_research_v4/results/confirmation",
}


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def save(path, obj):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    value = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    for attempt in range(30):
        try:
            temporary.write_text(value, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(min(.1 * (attempt + 1), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=JOBS, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = HERE / "results" / args.job
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE.json").exists():
        print("already_complete")
        return
    handle = (out / "worker.lock").open("a+b")
    handle.seek(0)
    if not handle.read(1):
        handle.write(b"0"); handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        print("already_running")
        return
    if args.worker:
        try:
            src = JOBS[args.job]
            print("POSTPROCESSING WORKER", args.job, stamp(), flush=True)
            while not (src / "DONE.json").exists() or (
                args.job != "replay_analysis" and not (src / "analysis/DONE.json").exists()
            ):
                save(out / "progress.json", dict(status="waiting_for_complete_campaign",
                    source=str(src), pid=os.getpid(), updated_at=stamp()))
                time.sleep(30)
            save(out / "progress.json", dict(status="rendering", source=str(src),
                 pid=os.getpid(), updated_at=stamp()))
            source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in HERE.glob("*.py")}
            save(out / "render_sources.json", source_hashes)
            if args.job == "terrain_controls":
                from terrain_controls import run
            elif args.job == "objective_witnesses":
                from objective_witnesses import run
            elif args.job == "cc_contract_audit":
                from cc_contract_audit import run
            elif args.job == "encoding_replay":
                from encoding_replay import run
            elif args.job == "scene_coverage":
                from scene_coverage import run
            elif args.job == "replay_analysis":
                from replay_analysis import run
            elif args.job == "confirmation_budget_diagnostics":
                from confirmation_budget_diagnostics import run
            elif args.job == "terrain_descriptive":
                from terrain_descriptive import run
            elif args.job == "terrain_layouts_verified":
                from terrain_layouts_verified import run
            elif args.job == "confirmation_metric_review":
                from confirmation_metric_review import run
            else:
                from campaign_figures import run
            result = run(src, out)
            result["finished_at"] = stamp()
            save(out / "DONE.json", result)
            save(out / "progress.json", dict(status="complete", pid=os.getpid(), updated_at=stamp()))
            print("POSTPROCESSING COMPLETE", flush=True)
        except Exception:
            save(out / ("ERROR_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"),
                 dict(traceback=traceback.format_exc(), failed_at=stamp()))
            raise
        finally:
            handle.close()
        return
    handle.close()
    logfile = out / ("worker_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    with logfile.open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
            "--job", args.job, "--worker"], cwd=HERE, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    info = dict(pid=process.pid, log=str(logfile), output=str(out), source=str(JOBS[args.job]),
                started_at=stamp(), purpose="postprocessing only")
    save(out / "launch.json", info)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
