"""Detached jobs with worker-held locks; all logs retained."""
from pathlib import Path
import argparse,datetime,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument("--job",choices=["audit","pilot"],required=True);args=p.parse_args()
    out=HERE/"results"/args.job;out.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (out/"worker.lock").open("a+b") as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b"0");lock.flush()
        lock.seek(0)
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:print("already_running");return
    if (out/"DONE.json").exists():print("already_complete");return
    if args.job=="pilot":
        audit=json.loads((HERE/"results/audit/DONE.json").read_text(encoding="utf-8"))
        assert audit["status"]=="complete" and audit["checks_passed"]
    log=out/("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    env=os.environ.copy();env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUNBUFFERED="1",MPLBACKEND="Agg")
    with log.open("ab",buffering=0) as f:
        proc=subprocess.Popen([sys.executable,"-u",str(HERE/(args.job+".py"))],cwd=HERE,env=env,
                              stdin=subprocess.DEVNULL,stdout=f,stderr=f,
                              creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP,close_fds=True)
    info=dict(pid=proc.pid,log=str(log),output=str(out),started_at=datetime.datetime.now().astimezone().isoformat())
    (out/"launch.json").write_text(json.dumps(info,indent=2),encoding="utf-8")
    print(json.dumps(info))
if __name__=="__main__":main()
