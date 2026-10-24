"""Launch/resume a detached Windows worker; existing outputs are preserved."""
import bootstrap
import argparse
import datetime
import json
import msvcrt
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['verify', 'experiment'])
    ap.add_argument('--config', default=str(bootstrap.ROOT / 'config.json'))
    args = ap.parse_args()
    config = Path(args.config).resolve()
    cfg = json.loads(config.read_text(encoding='utf-8'))
    directory = bootstrap.ROOT / 'results' / cfg['experiment_id']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / (args.mode + '_DONE.json')).exists():
        raise SystemExit('Already complete; preserved existing results. No duplicate launch.')
    with (directory / 'worker.lock').open('a+b') as guard:
        guard.seek(0)
        try:
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit('Worker active; duplicate launch blocked.')
        msvcrt.locking(guard.fileno(), msvcrt.LK_UNLCK, 1)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logs = directory / 'logs'
    logs.mkdir(exist_ok=True)
    stdout = logs / f'{args.mode}_{stamp}.out.log'
    stderr = logs / f'{args.mode}_{stamp}.err.log'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        proc = subprocess.Popen([sys.executable, '-u', str(bootstrap.ROOT / 'worker.py'),
                                 '--mode', args.mode, '--config', str(config)],
                                cwd=str(bootstrap.ROOT), stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                                close_fds=True)
    result = {'pid': proc.pid, 'mode': args.mode, 'started_at': stamp,
              'stdout': str(stdout), 'stderr': str(stderr), 'result_dir': str(directory)}
    (directory / f'launch_{args.mode}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
