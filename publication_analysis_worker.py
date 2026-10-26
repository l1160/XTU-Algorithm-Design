"""Detached logging wrapper for saved-result reanalysis."""
from pathlib import Path
import argparse,datetime,json,os,sys,traceback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    sys.stdin=open(os.devnull)
    sys.stdout=sys.stderr=(output/('analysis_'+stamp+'.log')).open('a',encoding='utf-8',buffering=1)
    print(datetime.datetime.now().astimezone().isoformat(),'SAVED ANALYSIS START',os.getpid(),flush=True)
    sys.argv=[sys.argv[0],'--output',str(output)]
    try:
        from reanalyze_confirmation import main as analyze
        analyze()
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        (output/('FAILED_'+stamp+'.json')).write_text(json.dumps(dict(error=error),ensure_ascii=False,indent=2),encoding='utf-8')
        raise


if __name__=='__main__':main()
