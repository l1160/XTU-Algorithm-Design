"""Scheduled hidden worker with fresh append-only logs and detached I/O."""
from pathlib import Path
import argparse
import datetime
import json
import os
import runpy
import sys
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    args=parser.parse_args()
    job=json.loads(Path(args.job).read_text(encoding="utf-8-sig"))
    root=Path(__file__).resolve().parent.parent
    output=Path(job["output"]).resolve();output.mkdir(parents=True,exist_ok=True)
    os.chdir(root)
    sys.path.insert(0,str(root))
    if job.get("dependency_path"):
        sys.path.insert(0,str(Path(job["dependency_path"]).resolve()))
    os.environ.update(WINDFARM_V6_BACKGROUND="1",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1",MPLBACKEND="Agg",PYTHONUTF8="1")
    sys.stdin=open(os.devnull)
    stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with (output/("worker_"+stamp+".log")).open("a",encoding="utf-8",buffering=1) as stream:
        sys.stdout=sys.stderr=stream
        print(datetime.datetime.now().astimezone().isoformat(),"WORKER START",os.getpid(),flush=True)
        try:
            if job["mode"]=="Check":
                sys.argv=["preflight","--output",str(output)]
                runpy.run_module("windfarm_research_v6.preflight",run_name="__main__")
            elif job["mode"]=="Run":
                sys.argv=["run","--config",job["config"],"--output",str(output)]
                runpy.run_module("windfarm_research_v6.run",run_name="__main__")
            else:
                raise ValueError("Unknown worker mode")
        except BaseException:
            traceback.print_exc()
            raise


if __name__=="__main__":main()
