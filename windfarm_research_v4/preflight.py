"""Deterministic adapter and checkpoint contracts on a synthetic selection task."""
import bootstrap_v4
import numpy as np
import dill
import algorithms as A
from model import repair_keys
from variants import Engine,VARIANTS,configure

class ContractFarm:
    """Synthetic objectives only; never reported as wind-farm evidence."""
    def __init__(self,options=None):
        self.options=dict(options or {});self.m=240;self.n=40;self.records=[];self.cache={}
        ids=np.arange(self.m)
        self.conflict=(abs(ids[:,None]-ids[None,:])==1)
    def groups(self,rng,mode,size=100):
        order=rng.permutation(self.m)
        groups=2 if self.options.get("groups")==2 else int(np.ceil(self.m/size))
        return list(np.array_split(order,groups))
    def evaluate(self,keys):
        repaired,selected,pairs,ok=repair_keys(keys,self.conflict,self.n)
        packed=selected.astype("<u2").tobytes()
        cv=float(self.conflict[np.ix_(selected,selected)].sum()/2)
        s=selected/self.m
        f=np.array([np.mean(s*s),np.mean((1-s)**2),np.mean((s-.5)**2)])
        self.records.append((packed,f.copy(),cv))
        return (repaired if self.options.get("writeback",True) else np.clip(keys,0,1).copy()),f,cv

def records_equal(a,b):
    return len(a)==len(b) and all(x[0]==y[0] and np.array_equal(x[1],y[1]) and x[2]==y[2] for x,y in zip(a,b))
def run():
    checks=[]
    for variant in ["NSGA3_base","CCwake_base","CCrandom_base","LM_base","DGEA_base"]:
        options=VARIANTS[variant];configure(options)
        a=A.Engine(ContractFarm(options),options["method"],937,600)
        b=Engine(ContractFarm(options),options["method"],937,600)
        while a.remaining:a.step()
        while b.remaining:b.step()
        assert records_equal(a.farm.records,b.farm.records),variant
        checks.append(dict(check="adapter baseline exact request sequence",variant=variant,passed=True))
        print("PREFLIGHT BASELINE",variant,flush=True)
    for variant,options in VARIANTS.items():
        configure(options);a=Engine(ContractFarm(options),options["method"],718,650)
        while len(a.farm.records)<300:a.step()
        blob=dill.dumps(a)
        while a.remaining:a.step()
        b=dill.loads(blob)
        while b.remaining:b.step()
        assert records_equal(a.farm.records,b.farm.records),variant
        assert len(a.farm.records)==650 and all(r[2]==0 and np.isfinite(r[1]).all() for r in a.farm.records)
        checks.append(dict(check="exact checkpoint continuation and finite feasible outputs",variant=variant,passed=True))
        print("PREFLIGHT RESUME",variant,flush=True)
    configure({})
    return dict(passed=True,checks=checks,scope="synthetic algorithm adapter contract; no wind-optimization evaluations")
