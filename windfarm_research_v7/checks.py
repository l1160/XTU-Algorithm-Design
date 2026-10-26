"""Necessary implementation checks for this single integrated revision."""
from pathlib import Path
import json
import pickle
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree

from windfarm_research_v6.model import OverlapFarm
from windfarm_research_v6.compat import legacy as A
from windfarm_research_v6.feasible import selected
from windfarm_research_v6.engines import layout_tokens
from windfarm_research_v6.lifecycle import save_json,now
from .model import FastFarm,dense_prim_length,sparse_writeback
from .engines import make_engine
from .runner import run_one


class Checks(unittest.TestCase):
    def farm(self,n=12,m=64,pop=8):
        return FastFarm("obs_00",m,n,initial_size=pop)

    def test_dense_MST_matches_independent_sparse_algorithm(self):
        rng=np.random.default_rng(20)
        for n in (2,8,32,100,401):
            xy=rng.random((n,3))*10000
            distances=cdist(xy,xy)
            expected=float(minimum_spanning_tree(distances).sum())
            self.assertAlmostEqual(dense_prim_length(distances)/expected,1.,places=12)

    def test_sparse_writeback_and_neutral_states(self):
        rng=np.random.default_rng(21)
        for mode in range(4):
            for _ in range(80):
                scores=rng.random(30) if mode==0 else np.clip(rng.normal(mode-1,.6,30),0,1)
                ids=np.sort(rng.choice(30,8,replace=False))
                keys=sparse_writeback(scores,ids)
                np.testing.assert_array_equal(selected(keys,8),ids)
                self.assertTrue(((keys>=0)&(keys<=1)).all())
                if mode==0:
                    raw=selected(scores,8)
                    unchanged=np.setdiff1d(np.arange(30),np.setxor1d(raw,ids))
                    np.testing.assert_array_equal(keys[unchanged],scores[unchanged])
                np.testing.assert_array_equal(sparse_writeback(scores,selected(scores,8)),scores)

    def test_exact_model_matches_frozen_objectives(self):
        rng=np.random.default_rng(22)
        for terrain in ("flat","ridge"):
            new=FastFarm("obs_00",64,12,terrain_kind=terrain,initial_size=8)
            old=OverlapFarm("obs_00",64,12,terrain_kind=terrain,initial_size=8)
            for _ in range(4):
                keys=rng.random(64)
                a=new.evaluate(keys);b=old.evaluate(keys)
                np.testing.assert_array_equal(a[0],b[0])
                np.testing.assert_allclose(new.records[-1][1],old.records[-1][1],rtol=1e-12,atol=1e-7)

    def test_legal_wide_steps_are_not_truncated(self):
        farm=self.farm(n=20,m=128,pop=16);rng=np.random.default_rng(23)
        a,_,_=farm.evaluate(rng.random(128));b,_,_=farm.evaluate(rng.random(128))
        parent=selected(a,20);target=selected(b,20)
        self.assertGreater(20-len(np.intersect1d(parent,target)),8)
        x,_,cv=farm.evaluate_operator(b,parent)
        np.testing.assert_array_equal(selected(x,20),target);self.assertEqual(cv,0)

    def test_preserves_DGEA_preselection(self):
        farm=self.farm(n=20,m=128,pop=16)
        engine=make_engine(farm,"DGEA_problem",24,budget=200,population=16)
        engine._dgea_step()
        original=A.dgea_pre;captured=[]
        def remember(*args,**kwargs):
            p,r=original(*args,**kwargs);captured.append(layout_tokens(p,20));return p,r
        with patch.object(A,"dgea_pre",side_effect=remember):engine._dgea_step()
        self.assertEqual(layout_tokens(engine.pop,20),captured[-1])

    def test_four_engines_shared_start_restart_and_budget(self):
        first=None
        for method in ("LMOCSO","LMOCSO_problem","DGEA","DGEA_problem"):
            farm=self.farm(n=20,m=128,pop=16)
            engine=make_engine(farm,method,25,budget=127,population=16)
            prefix=[r[0] for r in farm.records[:16]]
            if first is None:first=prefix
            else:self.assertEqual(prefix,first)
            engine.step();resumed=pickle.loads(pickle.dumps(engine))
            while engine.remaining:engine.step()
            while resumed.remaining:resumed.step()
            self.assertEqual(len(engine.farm.records),127)
            self.assertEqual([r[0] for r in engine.farm.records],[r[0] for r in resumed.farm.records])
            np.testing.assert_allclose(np.array([r[1] for r in engine.farm.records]),np.array([r[1] for r in resumed.farm.records]),rtol=1e-12,atol=1e-7)
            self.assertTrue(all(r[2]==0 for r in engine.farm.records))
            if method.endswith("_problem"):
                self.assertGreater(engine.operator_counts["local_exact_requests"],0)
                self.assertGreater(engine.operator_counts["native_operator_requests"],engine.operator_counts["local_exact_requests"])

    def test_integrated_runner_resume(self):
        folder=Path(tempfile.mkdtemp(prefix="runner_",dir=OUTPUT))
        cfg=dict(method="LMOCSO_problem",candidates=64,turbines=12,population=8,requests=33,seed=26,checkpoint_requests=250)
        self.assertEqual(run_one(cfg,folder,stop_after_steps=1)["status"],"paused")
        done=run_one(cfg,folder)
        self.assertEqual(done["requests"],33)
        self.assertTrue(done["independent_pareto_geometry_passed"])


def run(output):
    global OUTPUT
    OUTPUT=Path(output);OUTPUT.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    value=dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
               finished_at=now(),scope="Implementation checks only; no parameter sweep or performance claim",
               failures=[str(t) for t,_ in result.failures],errors=[str(t) for t,_ in result.errors])
    save_json(OUTPUT/("DONE.json" if result.wasSuccessful() else "FAILED.json"),value)
    if not result.wasSuccessful():raise RuntimeError("V7 implementation checks failed")
    return value
