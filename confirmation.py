"""Prepare portable background jobs for the frozen 180-run confirmation.

This command only prepares configuration. Start jobs with Start-Validation.ps1.
"""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parent


def save_new(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8-sig'))!=value:
            raise FileExistsError(f'Existing job differs; preserve it and use a new output directory: {path}')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def output_path(value,default):
    path=Path(value).resolve() if value else ROOT/default
    reference=ROOT/'reference_results'
    if path==ROOT or path==reference or reference in path.parents:
        raise ValueError('Reference evidence and the repository root cannot be experiment output directories')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('prepare-check',help='Prepare a job for the existing bounded implementation checks')
    check.add_argument('--output')
    prepare=sub.add_parser('prepare',help='Prepare the 180-run plan after the background checks pass')
    prepare.add_argument('--output')
    prepare.add_argument('--check-output')
    args=parser.parse_args()
    if args.command=='prepare-check':
        out=output_path(args.output,'verification/confirmation_public')
        path=out/'check_job.json'
        save_new(path,dict(mode='Check',output=str(out),dependency_path=''))
        print(json.dumps(dict(job=str(path),output=str(out),optimization_started=False),ensure_ascii=False,indent=2))
    else:
        from windfarm_validation.protocol import prepare as prepare_plan
        out=output_path(args.output,'reproduced/confirmation_public')
        check_out=Path(args.check_output).resolve() if args.check_output else ROOT/'verification/confirmation_public'
        job=prepare_plan(str(out),'',str(check_out))
        print(json.dumps(dict(job=str(out/'campaign_job.json'),output=job['output'],runs=180,
                             optimization_started=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
