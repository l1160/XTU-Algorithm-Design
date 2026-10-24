"""Single-factor mechanism variants, not claimed as original-paper replications."""
import bootstrap_v4
import numpy as np
import algorithms as A
from model_v4 import Farm
from pymoo.util.ref_dirs import get_reference_directions
ORIGINAL_EVALUATE=A.evaluate
ORIGINAL_APD=A.apd_ids

VARIANTS={
 "NSGA3_base":dict(method="NSGA3"),
 "CCwake_base":dict(method="CCGDE3_wake"),
 "CCrandom_base":dict(method="CCGDE3_random"),
 "CCrandom_2g40":dict(method="CCGDE3_random",groups=2),
 "CCrandom_2g80":dict(method="CCGDE3_random",groups=2,species=80),
 "LM_base":dict(method="LMOCSO"),
 "LM_no_writeback":dict(method="LMOCSO",writeback=False),
 "LM_keep_velocity":dict(method="LMOCSO",keep_velocity=True),
 "LM_range_APD":dict(method="LMOCSO",range_apd=True),
 "DGEA_base":dict(method="DGEA"),
 "DGEA_no_writeback":dict(method="DGEA",writeback=False),
 "DGEA_range_APD":dict(method="DGEA",range_apd=True),
}

def evaluate_keep_velocity(farm,x,v=None):
    x=np.atleast_2d(x)
    velocity=np.zeros_like(x) if v is None else v.copy()
    values=[farm.evaluate(row) for row in x]
    return A.Pop(np.array([r[0] for r in values]),np.array([r[1] for r in values]),np.array([r[2] for r in values]),velocity)

def range_apd(pop,refs,theta,ideal=None):
    origin=pop.f.min(axis=0) if ideal is None else ideal
    span=np.ptp(pop.f,axis=0)
    scaled=(pop.f-origin)/np.where(span>1e-14,span,1.)
    translated=A.Pop(pop.x,scaled,pop.cv,pop.v)
    return ORIGINAL_APD(translated,refs,theta,np.zeros(pop.f.shape[1]))

def configure(options):
    # Each job executes in its own process, and resets both hooks explicitly.
    A.evaluate=evaluate_keep_velocity if options.get("keep_velocity") else ORIGINAL_EVALUATE
    A.apd_ids=range_apd if options.get("range_apd") else ORIGINAL_APD

class DiagnosticFarm(Farm):
    def __init__(self,*args,options=None,**kwargs):
        self.options=dict(options or {})
        self.diagnostics=[]
        super().__init__(*args,**kwargs)
    def evaluate(self,keys):
        original=np.clip(np.asarray(keys,float),0,1)
        repaired,f,cv=super().evaluate(keys)
        self.diagnostics.append((float(np.mean(np.abs(repaired-original)>1e-12)),float(np.linalg.norm(repaired-original))))
        return (repaired if self.options.get("writeback",True) else original.copy()),f,cv
    def groups(self,rng,mode,size=100):
        if self.options.get("groups")==2:size=self.m//2
        return super().groups(rng,mode,size)

class Engine(A.Engine):
    def __init__(self,farm,method,seed,budget=2000,population=100):
        if not method.startswith("CCGDE3"):
            super().__init__(farm,method,seed,budget,population)
            return
        self.farm,self.method,self.budget,self.nominal=farm,method,budget,population
        self.rng=np.random.default_rng(seed)
        initial=self.rng.random((population,farm.m))
        self.refs=get_reference_directions("das-dennis",3,n_partitions=12)
        self.iteration=0;self.trace=[]
        self.pop=A.evaluate(farm,initial)
        self.group_list=farm.groups(self.rng,"wake" if method.endswith("wake") else "random")
        self.species=[self.pop.take(self.rng.permutation(population)[:farm.options.get("species",40)]) for _ in self.group_list]
        self.group_index=0;self.record()
    def record(self):
        super().record()
        p=self.algorithm.pop if self.method in ("NSGA2","NSGA3") else self.pop
        if p is None:return
        if self.method in ("NSGA2","NSGA3"):
            x=p.get("X");f=p.get("F");v=np.zeros_like(x)
        else:x,f,v=p.x,p.f,p.v
        self.trace[-1].update(population=len(x),genotypes_unique=len(np.unique(x,axis=0)),
                             nonzero_velocity_fraction=float(np.mean(np.abs(v)>1e-14)),
                             f_span=np.ptp(f,axis=0).tolist())
