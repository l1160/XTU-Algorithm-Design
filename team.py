"""Partition the frozen campaigns across team members without changing experiments."""
import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback
from reproduce import ROOT, setup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',choices=['repair','terrain'],required=True)
    parser.add_argument('--member',type=int,default=1)
    parser.add_argument('--members',type=int,default=7)
    parser.add_argument('--list',action='store_true')
    args=parser.parse_args()
    if not 1<=args.member<=args.members:
        parser.error('Require 1 <= member <= members')
    lifecycle,signature=setup(args.campaign)
    cfg=lifecycle.validated_config()
    jobs=lifecycle.specs(cfg)[args.member-1::args.members]
    if args.list:
        print(json.dumps({'member':args.member,'members':args.members,'runs':len(jobs),'jobs':jobs},indent=2))
        return
    if os.environ.get('WINDFARM_BACKGROUND_WORKER')!='1':
        parser.error('Use Start-TeamRun.ps1 to run in the background')
    out=lifecycle.OUT;out.mkdir(parents=True,exist_ok=True)
    from pilot import lock_file,now
    # All members use the same source signature and unchanged full campaign config.
    with lock_file(out/f'team_{args.member}_of_{args.members}.lock'):
        for attempt in range(120):
            try:
                gate=lock_file(out/'prepare.lock')
                break
            except OSError:
                time.sleep(.25)
        else:raise RuntimeError('Another process is preparing this campaign')
        with gate:
            current=signature()
            if (out/'manifest.json').exists():
                assert lifecycle.read(out/'manifest.json')==current,'Source changed; use a new clone/output'
            else:
                lifecycle.save(out/'manifest.json',current)
                lifecycle.save(out/'frozen_config.json',cfg)
        progress=out/f'team_{args.member}_of_{args.members}.json'
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            for spec in jobs:
                folder=out/spec['name'];folder.mkdir(exist_ok=True)
                if (folder/'DONE.json').exists():continue
                if (out/'PAUSE_REQUEST.json').exists():break
                lifecycle.save(progress,dict(status='running',current=spec['name'],pid=os.getpid(),updated_at=now()))
                stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                with (folder/f'worker_{stamp}.log').open('a',encoding='utf-8',buffering=1) as log:
                    with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                        try:lifecycle.worker(spec,cfg)
                        except Exception:
                            traceback.print_exc()
                            lifecycle.save(folder/f'ERROR_{stamp}.json',dict(error=traceback.format_exc(),failed_at=now()))
                            raise
            done=sum((out/s['name']/'DONE.json').exists() for s in jobs)
            lifecycle.save(progress,dict(status='complete' if done==len(jobs) else 'paused',completed=done,total=len(jobs),updated_at=now()))
        except Exception:
            lifecycle.save(progress,dict(status='error',error=traceback.format_exc(),updated_at=now()))
            raise
        finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
