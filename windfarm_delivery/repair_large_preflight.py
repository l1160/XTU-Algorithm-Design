"""LMOCSO/DGEA-only background preflight; all six policies use shared inputs."""
from pathlib import Path
import os
import sys
import json
import hashlib
import datetime
import traceback
import time
import msvcrt
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/repair_large_preflight"
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(ROOT / "windfarm_research_v4"))
import bootstrap_v4
from pilot import save, lock_file, now


def signature():
    source = ROOT / "windfarm_research_v4/results/confirmation/manifest.json"
    original = json.loads(source.read_text())
    for p, expected in original.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected, p
    paths = [Path(p) for p in original]
    paths += [Path(__file__), HERE / "repair_ablation_model.py", HERE / "repair_ablation_large_model.py",
              HERE / "repair_ablation_plan_large_scale.json", HERE / "research_scope_20261021.json"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def run():
    import numpy as np
    import dill
    import algorithms as A
    from variants import Engine, DiagnosticFarm
    from repair_ablation_model import decode
    from repair_ablation_large_model import RepairFarm, configure
    cfg = json.loads((HERE / "repair_ablation_plan_large_scale.json").read_text())
    assert cfg["algorithms"] == ["LMOCSO", "DGEA"]
    frozen = signature()
    if (OUT / "sources.json").exists():
        assert json.loads((OUT / "sources.json").read_text()) == frozen
    else:
        save(OUT / "sources.json", frozen)
    if (OUT / "DONE.json").exists():
        return
    checks = []

    def record(name, **details):
        checks.append(dict(name=name, passed=True, **details))
        save(OUT / "checks.json", checks)
        save(OUT / "progress.json", dict(stage=name, passed_checks=len(checks), pid=os.getpid(), updated_at=now()))
        print(now(), name, details, flush=True)

    # Reuse only independently checked decoding evidence, verifying its sources.
    prior = HERE / "results/repair_ablation_preflight"
    prior_hashes = json.loads((prior / "sources.json").read_text())
    for p, expected in prior_hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == expected
    prior_checks = json.loads((prior / "checks.json").read_text())
    required = {"exhaustive_graph_decode_equivalence", "backtrack_needed_witness", "ties_clipping_invalid_inputs", "real_case_baseline_output_and_scalar_cv"}
    assert required <= {r["name"] for r in prior_checks if r["passed"]}
    record("inherited_decode_evidence_verified", exhaustive_cases=6144, inherited_checks=sorted(required))

    # Independent sanity cases exercise the selection paths, not only fronts().
    refs = np.eye(3)
    x = np.arange(16.).reshape(4, 4)
    pop = A.Pop(x, np.array([[1, 1, 1], [-10, -10, -10], [-20, -20, -20], [-30, -30, -30.]]),
                np.array([0, .2, .1, .3]), np.zeros_like(x))
    chosen = A.lm_select(pop, refs, .5)
    assert len(chosen) and np.all(chosen.cv == 0)
    pre, ranks = A.dgea_pre(pop, refs, .5)
    assert np.any(pre.cv == 0)
    assert np.all(pre.cv[ranks == 1] == 0)
    assert [ids.tolist() for ids in A.fronts(pop)] == [[0], [2], [1], [3]]
    record("both_selection_paths_feasibility_priority")

    # Check the instrumentation wrapper against unmodified A.evaluate exactly.
    configure()
    f1 = RepairFarm("obs_01", 1600, 400, repair_mode="bounded_backtrack", shared_initial_requests=0)
    f2 = DiagnosticFarm("obs_01", 1600, 400)
    rng = np.random.default_rng(712)
    keys, velocity = rng.random((6, 1600)), rng.normal(size=(6, 1600))
    from variants import ORIGINAL_EVALUATE
    p1 = A.evaluate(f1, keys, velocity)
    p2 = ORIGINAL_EVALUATE(f2, keys, velocity)
    for attr in ("x", "f", "cv", "v"):
        np.testing.assert_array_equal(getattr(p1, attr), getattr(p2, attr))
    assert sum(d["velocity_coordinates_reset"] for d in f1.repair_details) > 0
    record("telemetry_does_not_change_evaluator_or_velocity")
    del f1, f2, p1, p2

    reference = None
    cases = []
    for method in cfg["algorithms"]:
        for mode in cfg["repair_modes"]:
            label = f"{method}_{mode}"
            checkpoint = OUT / (label + ".pkl")
            configure()
            if checkpoint.exists():
                with checkpoint.open("rb") as f:
                    engine = dill.load(f)
            else:
                farm = RepairFarm("obs_01", 1600, 400, repair_mode=mode)
                engine = Engine(farm, method, 712, budget=300, population=100)
                assert len(farm.records) == 100
                # LM may generate a small number of children per step. Never
                # assume one step gives 100 requests or a checkpoint at 200.
                while len(farm.records) < 200:
                    before = len(farm.records)
                    engine.step()
                    assert len(farm.records) > before
                assert 200 <= len(farm.records) < 300
                tmp = checkpoint.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    dill.dump(engine, f)
                tmp.replace(checkpoint)
            farm = engine.farm
            initial = dict(keys=np.array(farm.initial_keys), selected=[r[0] for r in farm.records[:100]],
                           metrics=np.array([r[1] for r in farm.records[:100]]), cv=np.array([r[2] for r in farm.records[:100]]))
            if reference is None:
                reference = initial
            for key in initial:
                np.testing.assert_array_equal(initial[key], reference[key])
            assert len(initial["keys"]) == 100 and np.all(initial["cv"] == 0)
            checkpoint_count = len(farm.records)
            while engine.remaining:
                before = len(engine.farm.records)
                engine.step()
                assert before < len(engine.farm.records) <= 300
            expected = [(r[0], r[1].tolist(), *r[2:]) for r in engine.farm.records]
            trace = engine.trace
            final_pop = engine.pop
            engine.farm.wake = None
            with checkpoint.open("rb") as f:
                resumed = dill.load(f)
            while resumed.remaining:
                resumed.step()
            actual = [(r[0], r[1].tolist(), *r[2:]) for r in resumed.farm.records]
            assert actual == expected and resumed.trace == trace
            for attr in ("x", "f", "cv", "v"):
                np.testing.assert_array_equal(getattr(resumed.pop, attr), getattr(final_pop, attr))
            assert len(actual) == 300
            for i, r in enumerate(resumed.farm.records):
                selected = np.frombuffer(r[0], dtype="<u2")
                assert len(selected) == 400
                feasible = not resumed.farm.conflict[np.ix_(selected, selected)].any()
                assert (r[2] == 0) == feasible
                assert np.isfinite(r[1]).all()
                detail = resumed.farm.repair_details[i]
                assert detail["shared_initialization"] == (i < 100)
                assert detail["mode"] == ("bounded_backtrack" if i < 100 else mode)
            case = dict(method=method, mode=mode, requests=300, checkpoint_requests=checkpoint_count,
                        initial_keys_sha256=hashlib.sha256(initial["keys"].tobytes()).hexdigest(),
                        initial_layouts_sha256=hashlib.sha256(b"".join(initial["selected"])).hexdigest(), exact_resume=True)
            cases.append(case)
            record("engine_contract_"+label, **case)
            del engine, resumed, farm, final_pop
    assert signature() == frozen
    save(OUT / "DONE.json", dict(passed=True, finished_at=now(), checks=checks, engine_cases=cases,
        source_hashes=frozen, scope="LMOCSO/DGEA functional preflight only; six short cases, seed712 outside campaign seeds; no performance ranking"))
    save(OUT / "progress.json", dict(stage="complete", passed_checks=len(checks), pid=os.getpid(), updated_at=now()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_file(OUT / "worker.lock")
    except OSError:
        return
    logpath = OUT / ("worker_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    log = logpath.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    sys.stdin = open(os.devnull)
    try:
        save(OUT / "launch.json", dict(pid=os.getpid(), started_at=now(), log=str(logpath)))
        print(now(), "START large-scale repair preflight", flush=True)
        run()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        save(OUT / ("ERROR_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".json"), dict(error=error, failed_at=now()))
        save(OUT / "progress.json", dict(stage="failed", pid=os.getpid(), updated_at=now()))
        raise
    finally:
        handle.close()
        log.flush()


if __name__ == "__main__":
    main()
