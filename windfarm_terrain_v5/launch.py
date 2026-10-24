"""Detached, locked, append-only-log execution of terrain work."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys,msvcrt,traceback,hashlib,time
import bootstrap_v5
ROOT=Path(__file__).resolve().parent
def stamp():return datetime.datetime.now().astimezone().isoformat()
def save(path,obj):
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    payload=json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)
    for attempt in range(30):
        try:
            tmp.write_text(payload,encoding="utf-8")
            tmp.replace(path)
            return
        except PermissionError:
            if attempt==29:raise
            time.sleep(min(.1*(attempt+1),1.))
# The reused worker also writes JSON via pilot.save. Only I/O behavior changes;
# its frozen numerical sources and request sequences remain unchanged.
import pilot as _shared_worker
_shared_worker.save=save
def main():
    p=argparse.ArgumentParser();p.add_argument("--job",choices=["terrain_audit","model_audit","terrain_figures","terrain_experiments","pipeline_check","dem_precision","dataset_package","io_check"],required=True)
    p.add_argument("--worker",action="store_true");args=p.parse_args()
    out=ROOT/"results"/args.job;out.mkdir(parents=True,exist_ok=True)
    handle=(out/"worker.lock").open("a+b");handle.seek(0)
    if not handle.read(1):handle.write(b"0");handle.flush()
    handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:print("already_running");return
    if (out/"DONE.json").exists():print("already_complete");return
    if args.worker:
        try:
            save(out/"progress.json",dict(status="starting",pid=os.getpid(),updated_at=stamp()))
            module=__import__(args.job)
            print("START",args.job,stamp(),flush=True)
            module.run(out)
        except Exception:
            save(out/("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"),
                 dict(traceback=traceback.format_exc(),failed_at=stamp()))
            raise
        finally:handle.close()
        return
    handle.close()
    logfile=out/("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    env=os.environ.copy();env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg",PYTHONUNBUFFERED="1")
    with logfile.open("ab",buffering=0) as log:
        child=subprocess.Popen([sys.executable,"-u",str(__file__),"--job",args.job,"--worker"],cwd=ROOT,env=env,
             stdin=subprocess.DEVNULL,stdout=log,stderr=log,close_fds=True,
             creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    info=dict(pid=child.pid,log=str(logfile),output=str(out),started_at=stamp())
    save(out/"launch.json",info);print(json.dumps(info))
if __name__=="__main__":main()
