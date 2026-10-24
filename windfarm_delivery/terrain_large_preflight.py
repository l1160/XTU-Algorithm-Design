"""Background gate for the declared three-method synthetic-ridge confirmation."""
from pathlib import Path
import os
import sys
import json
import hashlib
import datetime
import traceback
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results/terrain_large_preflight"
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path.insert(0, str(ROOT / "windfarm_terrain_v5"))
import bootstrap_v5
from pilot import lock_file, now
from repair_large_campaign import save


def signature():
    files = [Path(__file__), HERE / "terrain_large_plan.json", HERE / "repair_large_campaign.py"]
    for rel in ("windfarm_research_v4/results/confirmation/manifest.json", "windfarm_terrain_v5/results/terrain_experiments/manifest.json"):
        for p, expected in json.loads((ROOT / rel).read_text()).items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected, p
            files.append(Path(p))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def run():
    import numpy as np
    import dill
    from shapely.geometry import Point
    from model_terrain import TerrainFarm
    from variants import Engine, configure, VARIANTS
    cfg = json.loads((HERE / "terrain_large_plan.json").read_text())
    assert cfg["variants"] == ["CCwake_base", "LM_base", "DGEA_base"]
    frozen = signature()
    if (OUT / "sources.json").exists():
        assert json.loads((OUT / "sources.json").read_text()) == frozen
    else:
        save(OUT / "sources.json", frozen)
    if (OUT / "DONE.json").exists():
        return
    checks, engine_cases = [], []
    def record(name, **values):
        checks.append(dict(name=name, passed=True, **values))
        save(OUT / "checks.json", checks)
        save(OUT / "progress.json", dict(stage=name, passed_checks=len(checks), pid=os.getpid(), updated_at=now()))
        print(now(), name, values, flush=True)
    reference_xy = None
    reference_obstacles = None
    for case in cfg["cases"]:
        scenario = case["scenario"]
        farm = TerrainFarm(scenario, 1600, 400, terrain_kind="ridge", resolution=50., slope_limit_deg=15.)
        assert farm.scene.width == 7000 and farm.scene.height == 14000
        assert len(farm.certificate) >= 400
        assert all(not farm.terrain.excluded.covers(Point(p)) for p in farm.xy)
        assert farm.scene.site_mask(farm.xy, closed_obstacles=True).all()
        expected = np.load(ROOT / "windfarm_terrain_v5/data/ridge_50m/candidates_M1600.npy")
        np.testing.assert_array_equal(farm.xy, expected)
        if reference_xy is None:
            reference_xy = farm.xy.copy()
            reference_obstacles = farm.scene.obstacles.copy()
        np.testing.assert_array_equal(farm.xy, reference_xy)
        np.testing.assert_array_equal(farm.scene.obstacles, reference_obstacles)
        farm.ensure_ground()
        assert np.isfinite(farm.ground_dist).all()
        assert farm.ground_cache_metadata["order"] == json.loads((ROOT / "windfarm_terrain_v5/results/terrain_audit/surface_quadrature_orders.json").read_text())["ridge"]["order"]
        save(OUT / (scenario+"_model_metadata.json"), farm.case_metadata())
        record("scene_gate_"+scenario, certificate_size=len(farm.certificate), terrain_sha256=farm.terrain.digest,
               candidate_sha256=hashlib.sha256(farm.xy.tobytes()).hexdigest(), surface_cache=farm.ground_cache_metadata)
        if scenario == "obs_00":
            source = ROOT / "windfarm_terrain_v5/results/terrain_experiments/obs_00_ridge_M1600_N400_CCwake_base_s501/evaluations.npz"
            with np.load(source) as z:
                for i in (0, 100, 1999):
                    selected = z["selected"][i]
                    keys = np.linspace(.5, 1, 1600)
                    keys[selected] = np.linspace(0, .4, 400)
                    _, _, cv = farm.evaluate(keys)
                    assert cv == 0
                    np.testing.assert_array_equal(np.frombuffer(farm.records[-1][0], dtype="<u2"), selected)
                    np.testing.assert_allclose(farm.records[-1][1], z["metrics"][i], rtol=1e-12, atol=1e-8)
            record("saved_V5_objectives_reproduced", source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), requests=[0,100,1999])
        del farm

        # Wrapper captures returned initial keys without changing model objects
        # stored in checkpoints. It is active only for each engine's initialization.
        reference_initial = None
        for variant in cfg["variants"]:
            method = VARIANTS[variant]["method"]
            checkpoint = OUT / f"{scenario}_{variant}.pkl"
            configure({})
            if checkpoint.exists():
                with checkpoint.open("rb") as f:
                    payload = dill.load(f)
                engine = payload["engine"]
                initial_keys = payload["initial_keys"]
            else:
                farm = TerrainFarm(scenario, 1600, 400, terrain_kind="ridge", resolution=50., slope_limit_deg=15.)
                captured = []
                ordinary = farm.evaluate
                def capture(keys):
                    result = ordinary(keys)
                    captured.append(result[0].copy())
                    return result
                farm.evaluate = capture
                engine = Engine(farm, method, seed=812, budget=300, population=100)
                del farm.evaluate  # Restore class dispatch before any pickle.
                initial_keys = np.array(captured)
                assert len(farm.records) == len(initial_keys) == 100
                while len(farm.records) < 180:
                    before = len(farm.records)
                    engine.step()
                    assert len(farm.records) > before
                assert 180 <= len(farm.records) < 300
                tmp = checkpoint.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    dill.dump(dict(engine=engine, initial_keys=initial_keys), f)
                tmp.replace(checkpoint)
            farm = engine.farm
            initial = dict(keys=initial_keys, layouts=[r[0] for r in farm.records[:100]],
                           metrics=np.array([r[1] for r in farm.records[:100]]))
            if reference_initial is None:
                reference_initial = initial
            for key in initial:
                np.testing.assert_array_equal(initial[key], reference_initial[key])
            count_at_checkpoint = len(farm.records)
            while engine.remaining:
                before = len(farm.records)
                engine.step()
                assert before < len(farm.records) <= 300
            expected = [(r[0], r[1].tolist(), *r[2:]) for r in farm.records]
            trace = engine.trace
            farm.wake = None
            with checkpoint.open("rb") as f:
                resumed = dill.load(f)["engine"]
            while resumed.remaining:
                resumed.step()
            actual = [(r[0], r[1].tolist(), *r[2:]) for r in resumed.farm.records]
            assert actual == expected and resumed.trace == trace
            for r in resumed.farm.records:
                selected = np.frombuffer(r[0], dtype="<u2")
                assert len(selected) == 400 and r[2] == 0
                assert not resumed.farm.conflict[np.ix_(selected, selected)].any()
                assert np.isfinite(r[1]).all()
            item = dict(scenario=scenario, variant=variant, requests=300, checkpoint_requests=count_at_checkpoint,
                initial_keys_sha256=hashlib.sha256(initial_keys.tobytes()).hexdigest(), exact_resume=True)
            engine_cases.append(item)
            record("engine_gate_"+scenario+"_"+variant, **item)
            del engine, resumed, farm
    assert signature() == frozen
    save(OUT / "DONE.json", dict(passed=True, source_hashes=frozen, checks=checks, engine_cases=engine_cases,
        scope="Synthetic ridge, obs_00/obs_01, 3 declared large-scale adaptations. Functional gates only; seed812 outside future confirmation seeds.", finished_at=now()))
    save(OUT / "progress.json", dict(stage="complete", passed_checks=len(checks), pid=os.getpid(), updated_at=now()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_file(OUT / "worker.lock")
    except OSError:
        return
    logpath = OUT / ("worker_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".log")
    log = logpath.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    sys.stdin = open(os.devnull)
    try:
        save(OUT / "launch.json", dict(pid=os.getpid(), started_at=now(), log=str(logpath)))
        print(now(), "START terrain large-scale preflight", flush=True)
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
