"""Portable entry: relocate historic source manifests in memory, never rewrite evidence."""
from pathlib import Path,PureWindowsPath
import argparse
import hashlib
import json
import os
import sys
ROOT=Path(__file__).resolve().parent
ORIGINAL=PureWindowsPath("D:/suanfa")
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",MPLBACKEND="Agg")
sys.path.insert(0,str(ROOT/"windfarm_delivery"))
sys.path.insert(0,str(ROOT/"windfarm_terrain_v5"))


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def relocated(path):
    relative=PureWindowsPath(path)
    if relative.is_absolute():
        relative=relative.relative_to(ORIGINAL)
    target=ROOT.joinpath(*relative.parts).resolve()
    assert target.is_relative_to(ROOT),"Source manifest escaped package root"
    return target


def setup(campaign):
    import bootstrap_v5
    import repair_large_campaign as lifecycle
    output=ROOT/"reproduced"/campaign
    assert output.resolve().is_relative_to(ROOT)
    assert output!=ROOT/"windfarm_delivery/results/repair_large_campaign"
    assert output!=ROOT/"windfarm_delivery/results/terrain_large_campaign"
    preflight_name="repair_large_preflight" if campaign=="repair" else "terrain_large_preflight"
    preflight=ROOT/"windfarm_delivery/results"/preflight_name/"DONE.json"
    evidence=json.loads(preflight.read_text(encoding="utf-8"));assert evidence["passed"]

    def signature():
        result={}
        for historic,expected in evidence["source_hashes"].items():
            path=relocated(historic)
            assert digest(path)==expected,f"Packaged source differs: {path}"
            result[str(path)]=expected
        extra=[Path(__file__),preflight,ROOT/"requirements_research.txt",ROOT/"team.py",ROOT/"Start-TeamRun.ps1",ROOT/"Start-Reproduction.ps1"]
        if campaign=="repair":
            extra += [ROOT/"windfarm_delivery"/p for p in ("repair_large_campaign.py","repair_large_config.json","repair_large_analysis.py")]
        else:
            extra += [ROOT/"windfarm_delivery"/p for p in ("terrain_large_campaign.py","terrain_large_model.py","terrain_large_analysis.py")]
        for path in extra:result[str(path)]=digest(path)
        return result

    if campaign=="repair":
        original_config=lifecycle.validated_config
        def config():
            value=original_config();value["output"]=str(output)
            value["reproduction_scope"]="Relocated source and input reproduction; original results read-only"
            return value
        lifecycle.validated_config=config
    else:
        import terrain_large_campaign as terrain
        import repair_large_analysis as dispatch
        from terrain_large_analysis import run
        dispatch.run=run
        terrain.OUT=output;terrain.signature=signature
        original_config=terrain.config
        def config():
            value=original_config();value["output"]=str(output)
            value["reproduction_scope"]="Relocated source and input reproduction; original results read-only"
            return value
        lifecycle.validated_config=config
        lifecycle.specs=terrain.specs
        lifecycle.worker=terrain.worker
    lifecycle.OUT=output
    lifecycle.signature=signature
    lifecycle.__file__=str(Path(__file__).resolve())
    os.environ["WINDFARM_REPRO_CAMPAIGN"]=campaign
    return lifecycle,signature


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--campaign",choices=("repair","terrain"),default=os.environ.get("WINDFARM_REPRO_CAMPAIGN"))
    parser.add_argument("--run")
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.campaign is None:parser.error("Choose --campaign repair or terrain")
    lifecycle,signature=setup(args.campaign)
    cfg=lifecycle.validated_config();jobs=lifecycle.specs(cfg)
    assert len(jobs)==60 and cfg["evaluations"]==10000
    assert all("NSGA" not in s["name"] for s in jobs)
    fingerprint=signature()
    if args.dry_run:
        print(json.dumps(dict(campaign=args.campaign,runs=len(jobs),requests_per_run=cfg["evaluations"],
            output=str(lifecycle.OUT),verified_source_files=len(fingerprint),
            original_paths_not_opened=True,optimization_started=False),ensure_ascii=False))
        return
    # Full runs must enter through the scheduled launcher. Child workers inherit
    # this explicit marker; a terminal command cannot accidentally run a campaign.
    if os.environ.get("WINDFARM_BACKGROUND_WORKER")!="1":
        parser.error("Use Start-Reproduction.ps1 for background runs, or --dry-run to inspect the plan")
    sys.argv=[str(Path(__file__).resolve())]+(["--run",args.run] if args.run else [])
    lifecycle.main()


if __name__=="__main__":main()
