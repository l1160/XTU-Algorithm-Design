"""Only necessary harness/equivalence tests; no tuning or performance sweep."""
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from windfarm_research_v6.engines import ProblemEngine
from windfarm_research_v6.lifecycle import FileLock, now
from windfarm_research_v7.model import FastFarm
from .runtime import METHODS, make_engine, run_one, save_json, read

OPTIONS = dict(layout_diversity=True, compound=True, max_attempts=24)
OUTPUT = None


class Checks(unittest.TestCase):
    def farm(self):
        return FastFarm("obs_00", 64, 12, initial_size=8)

    def test_six_methods_common_initialization_and_full_state_resume(self):
        prefix = None
        for method in METHODS:
            engine = make_engine(self.farm(), method, 1101, 97, 8, **(OPTIONS if method.endswith("_problem") else {}))
            initial = [r[0] for r in engine.farm.records[:8]]
            if prefix is None:
                prefix = initial
            self.assertEqual(initial, prefix)
            engine.step()
            resumed = pickle.loads(pickle.dumps(engine))
            while engine.remaining:
                engine.step()
            while resumed.remaining:
                resumed.step()
            self.assertEqual(len(engine.farm.records), 97)
            self.assertEqual([r[0] for r in engine.farm.records], [r[0] for r in resumed.farm.records])
            np.testing.assert_allclose(np.array([r[1] for r in engine.farm.records]),
                                       np.array([r[1] for r in resumed.farm.records]), rtol=1e-12, atol=1e-7)
            self.assertTrue(all(r[2] == 0 for r in engine.farm.records))
            self.assertTrue(all("elapsed_cpu_seconds" in t for t in engine.trace))

    def test_cc_timing_does_not_change_frozen_algorithm(self):
        original = ProblemEngine(self.farm(), "CCGDE3", 1102, 137, 8, **OPTIONS)
        wrapped = make_engine(self.farm(), "CCGDE3_problem", 1102, 137, 8, **OPTIONS)
        while original.remaining:
            original.step()
        while wrapped.remaining:
            wrapped.step()
        self.assertEqual([r[0] for r in original.farm.records], [r[0] for r in wrapped.farm.records])
        np.testing.assert_array_equal(np.array([r[1] for r in original.farm.records]),
                                      np.array([r[1] for r in wrapped.farm.records]))

    def test_full_checkpoint_routing_and_refuse_changed_config(self):
        for method in ("CCGDE3_problem", "LMOCSO_problem", "DGEA_problem"):
            folder = Path(tempfile.mkdtemp(prefix=method+"_", dir=OUTPUT))
            config = dict(method=method, candidates=64, turbines=12, population=8, requests=53, seed=1103)
            self.assertEqual(run_one(config, folder, stop_after_steps=1)["status"], "paused")
            self.assertEqual(run_one(config, folder)["requests"], 53)
            with self.assertRaises(ValueError):
                run_one(dict(config, seed=1104), folder)

    def test_json_retries_sharing_violation_and_duplicate_lock(self):
        from . import runtime
        folder = Path(tempfile.mkdtemp(prefix="io_", dir=OUTPUT))
        real = runtime.os.replace
        calls = []
        def sometimes(a, b):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("simulated reader")
            real(a, b)
        with patch.object(runtime.os, "replace", side_effect=sometimes):
            save_json(folder/"state.json", {"status": "complete"})
        self.assertEqual(read(folder/"state.json"), {"status": "complete"})
        with FileLock(folder/"run.lock"):
            with self.assertRaises(RuntimeError):
                with FileLock(folder/"run.lock"):
                    pass

    def test_independent_hv_front_and_exact_sign_flip(self):
        from .analysis import nondominated, volume3, exact_p, holm
        from pymoo.indicators.hv import HV
        from windfarm_research_v6.compat import legacy as A
        rng = np.random.default_rng(20261024)
        points = rng.random((300, 3))
        front = nondominated(points)
        expected = points[A.NDS.do(points, only_non_dominated_front=True)]
        self.assertEqual({tuple(p) for p in front}, {tuple(p) for p in expected})
        self.assertAlmostEqual(volume3(front, np.full(3, 1.1)), HV(ref_point=np.full(3, 1.1))(points), places=12)
        self.assertAlmostEqual(exact_p(np.ones(10)), 2/1024)
        self.assertEqual(exact_p(np.zeros(10)), 1.)
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])


def run(output):
    global OUTPUT
    OUTPUT = Path(output)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    value = dict(status="passed" if result.wasSuccessful() else "failed", tests=result.testsRun,
                 finished_at=now(), scope="Unified runtime equivalence, complete restart, independent statistics, and durable I/O; no performance inference",
                 failures=[str(t) for t,_ in result.failures], errors=[str(t) for t,_ in result.errors])
    save_json(OUTPUT/("DONE.json" if result.wasSuccessful() else "FAILED.json"), value)
    if not result.wasSuccessful():
        raise RuntimeError("Validation harness checks failed")
