"""Bounded functional checks, not performance replicates or a ranking study."""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad

from .compat import ROOT, legacy as A, BASELINES, METHODS
from .engines import make_engine, diverse_select, layout_tokens, ProblemEngine
from .feasible import encode, selected, token, validate, project
from .model import OverlapFarm
from .wake import overlap_fraction, wake_squared
from .lifecycle import FileLock, save_json, now
from .run import run_one, validate_config


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = Path(tempfile.mkdtemp(prefix="checks_",dir=OUTPUT))
        cls.common_initial = []

    def farm(self,terrain="flat",m=128,n=20,pop=16,incremental=True):
        return OverlapFarm("obs_00",m,n,terrain_kind=terrain,initial_size=pop,
                           cache_root=self.folder/"cache",incremental=incremental)

    def test_overlap_against_independent_quadrature(self):
        for r,w,d in [(38.5,61.6,61.59),(38.5,61.6,61.61),(38.5,61.6,99.9),
                      (2.,1.,.4),(2.,1.,1.5),(2.,3.,0.),(2.,3.,5.),(2.,3.,2.)]:
            lo,hi = max(-r,d-w),min(r,d+w)
            if lo>=hi:
                expected=0.
            else:
                x=(r*r-w*w+d*d)/(2*d) if d else 0.
                points=[x] if lo<x<hi else None
                area=quad(lambda v:2*min(math.sqrt(max(0.,r*r-v*v)),math.sqrt(max(0.,w*w-(v-d)**2))),
                          lo,hi,points=points,epsabs=1e-8,epsrel=1e-10,limit=200)[0]
                expected=area/(math.pi*r*r)
            self.assertAlmostEqual(float(overlap_fraction(r,w,d)),expected,places=8)

    def test_wake_edges_upstream_and_height(self):
        farm=self.farm(m=32,n=8)
        direction=np.array([math.cos(math.radians(7.5)),math.sin(math.radians(7.5))])
        normal=np.array([-direction[1],direction[0]])
        values=[]
        for lateral in (61.59,61.61):
            points=np.array([[1000.,1000.],np.array([1000.,1000.])+308*direction+lateral*normal])
            w=wake_squared(farm.scene,points)
            self.assertEqual(w[0,0,1],0.)
            self.assertTrue(np.all(np.diagonal(w,axis1=1,axis2=2)==0))
            values.append(float(np.sqrt(w[0,1,0])))
            self.assertEqual(wake_squared(farm.scene,points,[0,1000])[0,1,0],0.)
        self.assertTrue(all(.09<v<.10 for v in values))
        self.assertLess(abs(values[1]-values[0]),.001)

    def test_overlap_rejects_invalid_geometry(self):
        for args in [(-1,2,1),(1,2,-1),(1,np.nan,1)]:
            with self.assertRaises(ValueError):overlap_fraction(*args)

    def test_compound_exchange_and_fixed_outside(self):
        conflict=np.zeros((4,4),bool)
        conflict[2,[0,1]]=True;conflict[[0,1],2]=True
        scores=np.array([.8,.9,.01,.02]);anchor=np.array([0,1])
        p=project(scores,anchor,conflict,np.random.default_rng(2),max_changes=2)
        np.testing.assert_array_equal(p.indices,[2,3]);self.assertTrue(p.compound)
        p=project(scores,anchor,conflict,np.random.default_rng(2),group=[0,2,3],max_changes=2)
        self.assertIn(1,p.indices);self.assertNotIn(2,p.indices)
        self.assertTrue(validate(p.indices,conflict,2))

    def test_projection_bounded_failure_and_hash(self):
        conflict=np.zeros((4,4),bool)
        conflict[2:,:2]=True;conflict[:2,2:]=True
        p=project([.01,.02,.8,.9],np.array([0,1]),conflict,np.random.default_rng(4),max_changes=1,max_attempts=4)
        self.assertEqual(p.changed,0);self.assertLessEqual(p.attempts,4)
        self.assertTrue(p.primary_duplicate)
        self.assertGreaterEqual(p.duplicate_rejections,1)
        self.assertEqual(token([1,0]),token([0,1]))

    def test_projection_random_constraints(self):
        rng=np.random.default_rng(42)
        xy=np.arange(30.)[:,None]
        conflict=np.abs(xy-xy.T)<1.5;np.fill_diagonal(conflict,False)
        parent=np.arange(0,30,3)
        for _ in range(80):
            group=rng.choice(30,20,replace=False)
            proposal=project(rng.random(30),parent,conflict,rng,group=group)
            self.assertTrue(validate(proposal.indices,conflict,10))
            np.testing.assert_array_equal(np.setdiff1d(parent,group),np.setdiff1d(proposal.indices,group))
            np.testing.assert_array_equal(selected(proposal.keys,10),proposal.indices)
            self.assertLessEqual(proposal.attempts,24)
            parent=proposal.indices

    def test_encoding_preserves_actual_decision(self):
        scores=np.zeros(12)
        ids=np.array([1,4,7,9])
        keys=encode(scores,ids)
        np.testing.assert_array_equal(selected(keys,4),ids)
        self.assertEqual(len(np.unique(keys)),12)

    def test_layout_api_does_not_round_invalid_indices(self):
        farm=self.farm(m=32,n=8,pop=8)
        scores=np.random.default_rng(2).random(32)
        keys,_,_=farm.evaluate(scores)
        ids=selected(keys,8)
        with self.assertRaises(ValueError):farm.evaluate_layout(ids.astype(float)+.2,scores)
        with self.assertRaises(ValueError):farm.physical_metrics(ids.astype(float)+.2)
        with self.assertRaises(ValueError):farm.evaluate_layout(ids,np.zeros(31))

    def test_selection_fills_from_dominated_distinct_layouts(self):
        x=np.array([encode(np.zeros(20),[(i+j)%20 for j in range(4)]) for i in range(12)])
        f=np.tile(np.arange(12.)[:,None],(1,3))
        p=A.Pop(x,f,np.zeros(12),np.zeros_like(x))
        pool=A.join(p,p)
        refs=A.get_reference_directions("das-dennis",3,n_partitions=3)
        result=diverse_select(pool,12,4,refs,.5,"LMOCSO")
        self.assertEqual(len(result),12)
        self.assertEqual(len(set(layout_tokens(result,4))),12)

    def test_incremental_matches_full_all_objectives(self):
        for terrain in ("flat","ridge"):
            farm=self.farm(terrain,m=64,n=12)
            rng=np.random.default_rng(31)
            xx,_,cv=farm.evaluate(rng.random(64));self.assertEqual(cv,0)
            anchor=selected(xx,12)
            for _ in range(5):
                p=project(rng.random(64),anchor,farm.conflict,rng,max_changes=3)
                _,_,cv=farm.evaluate_layout(p.indices,p.keys,parent=anchor)
                exact=farm.physical_metrics(p.indices,force_full=True)
                np.testing.assert_allclose(farm.records[-1][1],exact,rtol=1e-11,atol=1e-8)
                self.assertEqual(cv,0)
                anchor=p.indices
            self.assertGreater(farm.statistics["incremental_wake_evaluations"],0)

    def test_baseline_wrappers_preserve_frozen_trajectories(self):
        for family,old_name in BASELINES.items():
            a=self.farm();b=self.farm()
            wrapped=make_engine(a,family,77,budget=65,population=16)
            original=A.Engine(b,old_name,77,budget=65,population=16)
            while wrapped.remaining:wrapped.step()
            while original.remaining:original.step()
            self.assertEqual([r[0] for r in a.records],[r[0] for r in b.records])
            np.testing.assert_array_equal(np.array([r[1] for r in a.records]),np.array([r[1] for r in b.records]))

    def test_six_methods_shared_initialization_budget_and_resume(self):
        initial=[]
        for method in METHODS:
            farm=self.farm()
            engine=make_engine(farm,method,123,budget=97,population=16)
            initial.append(([r[0] for r in farm.records[:16]],np.array([r[1] for r in farm.records[:16]])))
            engine.step()
            restored=pickle.loads(pickle.dumps(engine,protocol=pickle.HIGHEST_PROTOCOL))
            while engine.remaining:engine.step()
            while restored.remaining:restored.step()
            self.assertEqual(len(engine.farm.records),97)
            self.assertEqual([r[0] for r in engine.farm.records],[r[0] for r in restored.farm.records])
            np.testing.assert_array_equal(np.array([r[1] for r in engine.farm.records]),np.array([r[1] for r in restored.farm.records]))
            self.assertEqual(engine.rng.bit_generator.state,restored.rng.bit_generator.state)
            self.assertTrue(all(r[2]==0 for r in engine.farm.records))
            if method.endswith("_problem"):
                counts=engine.operator_counts
                required={"CCGDE3_problem":"cooperative_differential_steps","LMOCSO_problem":"competitive_swarm_steps","DGEA_problem":"direction_guided_steps"}[method]
                self.assertGreater(counts[required],0)
                self.assertGreater(counts["offspring_proposals"],0)
                if method.startswith("CCGDE3"):
                    self.assertGreater(counts["context_parent_proposals"],0)
                else:
                    self.assertEqual(counts["context_parent_proposals"],0)
                pop=engine.archive if method.startswith("DGEA") else engine.pop
                self.assertEqual(len(pop),len(set(layout_tokens(pop,20))))
                if not method.startswith("CCGDE3"):self.assertEqual(len(pop),16)
        for keys,metrics in initial[1:]:
            self.assertEqual(keys,initial[0][0]);np.testing.assert_array_equal(metrics,initial[0][1])

    def test_run_checkpoint_guard_and_complete_artifacts(self):
        folder=self.folder/"runner"
        cfg=dict(method="LMOCSO_problem",candidates=64,turbines=12,population=8,requests=25,seed=333,checkpoint_requests=8)
        paused=run_one(cfg,folder,stop_after_steps=1)
        self.assertEqual(paused["status"],"paused")
        self.assertTrue((folder/"last.pkl").exists())
        with self.assertRaises(ValueError):run_one(dict(cfg,seed=334),folder)
        done=run_one(cfg,folder)
        self.assertEqual(done["requests"],25)
        self.assertEqual(done["status"],"complete")
        self.assertTrue(done["independent_pareto_geometry_passed"])
        self.assertEqual(run_one(cfg,folder)["finished_at"],done["finished_at"])
        for filename in ("evaluations.npz","trace.json","case_metadata.json","manifest.json","DONE.json"):
            self.assertTrue((folder/filename).exists())

    def test_progress_without_checkpoint_never_restarts(self):
        folder=self.folder/"orphan"
        folder.mkdir();save_json(folder/"progress.json",{"requests":12})
        with self.assertRaises(ValueError):run_one(dict(method="DGEA",candidates=32,turbines=8,population=8,requests=25),folder)

    def test_initialization_checkpoint_survives_first_step_failure(self):
        folder=self.folder/"initial_checkpoint"
        cfg=dict(method="LMOCSO_problem",candidates=64,turbines=12,population=8,
                 requests=25,seed=335,checkpoint_requests=250)
        with patch.object(ProblemEngine,"step",side_effect=RuntimeError("synthetic first-step interruption")):
            with self.assertRaisesRegex(RuntimeError,"synthetic first-step"):
                run_one(cfg,folder)
        with (folder/"last.pkl").open("rb") as stream:
            saved=pickle.load(stream)
        self.assertEqual(len(saved["engine"].farm.records),8)
        done=run_one(cfg,folder)
        self.assertEqual(done["status"],"complete")
        self.assertEqual(done["requests"],25)

    def test_config_validation_and_lock_exclusion(self):
        for value in [{"method":"NSGA3"},{"turbines":1600},{"requests":2},{"seed":True},{"compound":"false"}]:
            with self.assertRaises(ValueError):validate_config(value)
        lock=self.folder/"exclusive.lock"
        with FileLock(lock):
            with self.assertRaises(RuntimeError):
                with FileLock(lock):pass


class ProgressResult(unittest.TextTestResult):
    def stopTest(self,test):
        super().stopTest(test)
        save_json(OUTPUT/"progress.json",{"status":"running","tests_finished":self.testsRun,
                  "failures":len(self.failures),"errors":len(self.errors),"updated_at":now(),
                  "scope":"Bounded functional checks, no performance experiment"})


def main():
    global OUTPUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if os.environ.get("WINDFARM_V6_BACKGROUND") != "1":
        parser.error("Use Start-V6.ps1 -Mode Check for detached checks")
    OUTPUT=Path(args.output).resolve();OUTPUT.mkdir(parents=True,exist_ok=True)
    with FileLock(OUTPUT/"preflight.lock"):
        if (OUTPUT/"DONE.json").exists():
            raise RuntimeError("Completed check output already exists; choose a new directory")
        print(now(),"START FUNCTIONAL CHECKS",flush=True)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2,resultclass=ProgressResult).run(suite)
        value={"status":"passed" if result.wasSuccessful() else "failed","tests":result.testsRun,
               "failures":[str(t) for t,_ in result.failures],"errors":[str(t) for t,_ in result.errors],
               "finished_at":now(),"scope":"Mathematical, operator, baseline-preservation, and restart checks only",
               "performance_experiments":0}
        save_json(OUTPUT/("DONE.json" if result.wasSuccessful() else "FAILED.json"),value)
        save_json(OUTPUT/"progress.json",value)
        print(now(),"FUNCTIONAL CHECKS",value["status"],flush=True)
        if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
