"""Reuse the tested lifecycle in this process; frozen V6 files stay unchanged."""
from pathlib import Path
from windfarm_research_v6 import run as lifecycle
from windfarm_research_v6.compat import ROOT
from windfarm_research_v6.lifecycle import sha
from .model import FastFarm
from .engines import make_engine
from . import REVISION

ORIGINAL_SIGNATURE=lifecycle.signature


def signature(config):
    result=ORIGINAL_SIGNATURE(config)
    paths=list((ROOT/"windfarm_research_v7").glob("*.py"))
    launcher=ROOT/"Start-V7.ps1"
    if launcher.exists():paths.append(launcher)
    for path in paths:result["sources"][path.relative_to(ROOT).as_posix()]=sha(path)
    result["algorithm_revision"]=REVISION
    return result


def run_one(config,output,stop_after_steps=None):
    if config.get("method") not in ("LMOCSO","DGEA","LMOCSO_problem","DGEA_problem"):
        raise ValueError("Only LM/DGEA and their integrated improvements are in this revision")
    # Injection is local to the dedicated process, not a file edit or a change
    # to the algorithm functions used by either baseline.
    lifecycle.OverlapFarm=FastFarm
    lifecycle.make_engine=make_engine
    lifecycle.signature=signature
    return lifecycle.run_one(config,output,stop_after_steps=stop_after_steps)
