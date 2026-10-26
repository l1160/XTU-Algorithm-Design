"""One exact evaluator and full-state lifecycle for the six frozen algorithms."""
from pathlib import Path
import json
import os
import pickle
import time

from windfarm_research_v6 import run as base_run
from windfarm_research_v6 import lifecycle
from windfarm_research_v6.compat import ROOT
from windfarm_research_v6.engines import BaselineEngine, ProblemEngine, layout_tokens
from windfarm_research_v7.model import FastFarm
from windfarm_research_v7.engines import Baseline, Optimized
from . import REVISION

ORIGINAL_SIGNATURE = base_run.signature
METHODS = ("CCGDE3", "CCGDE3_problem", "LMOCSO", "LMOCSO_problem", "DGEA", "DGEA_problem")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def replace(temp, target):
    # Windows readers can temporarily hold a file without FILE_SHARE_DELETE.
    for attempt in range(201):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            if attempt == 200:
                raise
            time.sleep(.05)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    replace(temp, path)


def save_checkpoint(folder, payload):
    temp = folder / f"last.pkl.{os.getpid()}.tmp"
    with temp.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    replace(temp, folder / "last.pkl")


class TimedCC(ProblemEngine):
    """Logging only: inherited V6 CC operators, selection and RNG are unchanged."""
    def _details(self):
        BaselineEngine._details(self)
        wall, cpu = self.farm.elapsed()
        self.trace[-1].update(
            working_population=len(self.pop),
            working_physical_unique=len(set(layout_tokens(self.pop, self.farm.n))),
            archive_population=len(self.context_archive) if hasattr(self, "context_archive") else None,
            elapsed_wall_seconds=wall, elapsed_cpu_seconds=cpu)


def make_engine(farm, method, seed, budget=10000, population=100, **options):
    if method not in METHODS:
        raise ValueError("Unknown frozen method")
    family = method.removesuffix("_problem")
    if not method.endswith("_problem"):
        return Baseline(farm, family, seed, budget, population)
    if options != dict(layout_diversity=True, compound=True, max_attempts=24):
        raise ValueError("Confirmation uses one fixed design, without parameter tuning")
    if family == "CCGDE3":
        return TimedCC(farm, family, seed, budget, population, **options)
    return Optimized(farm, family, seed, budget, population)


def signature(config):
    result = ORIGINAL_SIGNATURE(config)
    paths = list((ROOT / "windfarm_research_v7").glob("*.py"))
    paths += list((ROOT / "windfarm_validation").glob("*.py"))
    paths += [ROOT / "Start-Validation.ps1", ROOT / "Read-Validation.ps1"]
    for path in sorted(paths):
        result["sources"][path.relative_to(ROOT).as_posix()] = lifecycle.sha(path)
    result.update(validation_revision=REVISION,
                  algorithm_versions={"CCGDE3_problem": "frozen V6", "LMOCSO_problem": "frozen V7", "DGEA_problem": "frozen V7"},
                  evaluation_backend="All six methods: V7 FastFarm, exact dense Prim, V6 rotor-overlap wake")
    return result


def run_one(config, output, stop_after_steps=None):
    # Process-local routing; no changes to archived V6/V7 source files.
    lifecycle.save_json = save_json
    base_run.save_json = save_json
    base_run.save_checkpoint = save_checkpoint
    base_run.OverlapFarm = FastFarm
    base_run.make_engine = make_engine
    base_run.signature = signature
    result = base_run.run_one(config, output, stop_after_steps=stop_after_steps)
    if result["status"] == "complete":
        result["result_scope"] = "One prespecified confirmation cell; compare only within the same case and seed"
        save_json(Path(output) / "DONE.json", result)
    return result
