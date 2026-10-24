"""Launch/resume detached Windows worker; lock protects against duplicates."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config.json'))
    parser.add_argument('--output', default=str(ROOT / 'results' / 'comparison_v3'))
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 18:
        parser.error('--workers must be between 1 and 18')
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Worker lock is authoritative even when stale PID metadata exists.
    import msvcrt
    with (directory / 'worker.lock').open('a+b') as lock:
        lock.seek(0)
        if (directory / 'worker.lock').stat().st_size == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print(json.dumps({'status': 'already_running', 'output': str(directory)}))
            return
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
    if (directory / 'DONE.json').exists():
        print(json.dumps({'status': 'already_complete', 'output': str(directory)}))
        return
    if (directory / 'PAUSE_REQUEST.json').exists():
        print(json.dumps({'status': 'pause_requested', 'output': str(directory), 'action': 'Explicit resume is required before clearing the pause request'}))
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path = directory / ('worker_' + stamp + '.log')
    parallel = (args.workers > 1 or (directory / 'parallel_manifest.json').exists()) and not args.validate_only
    entry = ROOT / ('parallel.py' if parallel else 'worker.py')
    cmd = [sys.executable, '-u', str(entry), '--config', str(Path(args.config).resolve()), '--output', str(directory)]
    if parallel:
        cmd.extend(['--workers', str(args.workers)])
    if args.validate_only:
        cmd.append('--validate-only')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
    with log_path.open('ab', buffering=0) as file:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=file, stderr=file,
                                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP, close_fds=True)
    info = {'pid': proc.pid, 'log': str(log_path), 'output': str(directory), 'config': str(Path(args.config).resolve()), 'launched_at': datetime.datetime.now().isoformat(), 'mode': 'parallel' if parallel else 'serial', 'workers': args.workers if parallel else 1}
    (directory / 'launch.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
