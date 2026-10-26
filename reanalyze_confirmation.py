"""Rebuild final statistics from the published saved records, without optimization."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=str(ROOT/'reference_results/confirmation_20261024'))
    parser.add_argument('--output',default=str(ROOT/'reproduced/confirmation_reanalysis'))
    args=parser.parse_args()
    source=Path(args.source).resolve();output=Path(args.output).resolve()
    reference=ROOT/'reference_results'
    if output==source or source in output.parents or output==ROOT or output==reference or reference in output.parents:
        raise ValueError('Write reanalysis into a separate output directory')
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
    from windfarm_research_v6.lifecycle import FileLock,now,sha
    from windfarm_validation.runtime import save_json
    from windfarm_validation.analysis import run
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(output/'reanalysis.lock'):
        if (output/'DONE.json').exists():
            print('Already complete; results preserved',flush=True);return
        plan=read(source/'plan.json')
        for relative,digest in plan['source_sha256'].items():assert sha(ROOT/relative)==digest,relative
        for cell in plan['jobs']:
            cell['output']=str(source/'runs'/cell['id'])
            cell['config']=str(source/'configs'/(cell['id']+'.json'))
            assert read(Path(cell['output'])/'DONE.json')['status']=='complete'
        plan['output']=str(output)
        for case in plan['cases']:
            dst=output/'preparation'/case['id'];dst.mkdir(parents=True,exist_ok=True)
            for name in ('geometry.npz','metadata.json','DONE.json'):
                src=source/'preparation'/case['id']/name;target=dst/name
                if target.exists():assert sha(src)==sha(target)
                else:shutil.copy2(src,target)
        save_json(output/'provenance.json',dict(source=str(source),source_plan_sha256=sha(source/'plan.json'),
            mode='Saved-result reanalysis, including the original objective consistency checks; no new optimization',started_at=now()))
        result=run(plan)
        save_json(output/'DONE.json',result)
        print(now(),'SAVED-RESULT REANALYSIS COMPLETE',flush=True)


if __name__=='__main__':main()
