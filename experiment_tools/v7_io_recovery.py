"""Retry Windows atomic status writes, retaining frozen algorithms/checkpoints.

This wrapper changes persistence only. It routes the remaining worker processes
through the same retry wrapper and records every retry in each output directory.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
job_path=Path(sys.argv[sys.argv.index("--job")+1]).resolve()
job=json.loads(job_path.read_text(encoding="utf-8-sig"))
sys.path.insert(0,str(ROOT))
if job.get("dependency_path"):sys.path.insert(0,job["dependency_path"])
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg")

from windfarm_research_v6 import lifecycle,run as run_module
from windfarm_research_v7 import worker

statistics={"atomic_replace_retries":0,"retry_sleep_seconds":0.}


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.retry.tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for attempt in range(101):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==100:raise
            statistics["atomic_replace_retries"]+=1
            statistics["retry_sleep_seconds"]+=.05
            time.sleep(.05)


lifecycle.save_json=save_json
run_module.save_json=save_json
original_popen=subprocess.Popen
original_worker=(ROOT/"windfarm_research_v7/worker.py").resolve()


def routed_popen(command,*args,**kwargs):
    command=list(command)
    if len(command)>1 and Path(command[1]).resolve()==original_worker:
        command[1]=str(Path(__file__).resolve())
    return original_popen(command,*args,**kwargs)


subprocess.Popen=routed_popen
try:
    worker.main()
finally:
    output=Path(job["output"])
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_json(output/("io_recovery_"+stamp+".json"),dict(statistics,
        scope="Atomic JSON-write retries only; frozen objective, algorithm, configuration and complete checkpoints unchanged",
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        finished_at=datetime.datetime.now().astimezone().isoformat()))
