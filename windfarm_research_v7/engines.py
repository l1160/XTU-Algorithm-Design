"""Concrete LM/DGEA optimization: preserve native search and add physical refinement."""
from collections import Counter
import time
import numpy as np
from scipy.spatial import ConvexHull

from windfarm_research_v6.compat import legacy as A
from windfarm_research_v6.engines import BaselineEngine, ProblemEngine as V6Problem, layout_tokens
from windfarm_research_v6.feasible import selected, token


def unique_pop(pop,n):
    keep={}
    for i,k in enumerate(layout_tokens(pop,n)):
        if k not in keep or pop.cv[i]<pop.cv[keep[k]]:keep[k]=i
    return pop.take(list(keep.values()))


class Baseline(BaselineEngine):
    def _details(self):
        super()._details()
        wall,cpu=self.farm.elapsed()
        self.trace[-1].update(working_population=len(self.pop),
                             working_physical_unique=len(set(layout_tokens(self.pop,self.farm.n))),
                             archive_population=len(self.archive) if hasattr(self,"archive") else None,
                             elapsed_wall_seconds=wall,elapsed_cpu_seconds=cpu)


class Optimized(Baseline):
    active_floor=20

    def __init__(self,farm,family,seed,budget=10000,population=100):
        self.operator_counts=Counter();self.local_mode=0
        super().__init__(farm,family,seed,budget,population)
        self.display_method=family+"_optimized_v7"
        initial=A.Pop(np.array(farm.initial_keys[:population]),
                      np.array([[-r[1][0],r[1][1],r[1][2]] for r in farm.records[:population]])/farm.scales,
                      np.array([r[2] for r in farm.records[:population]]),np.zeros((population,farm.m)))
        initial=initial.take(np.flatnonzero(initial.cv==0))
        if not len(initial):raise ValueError("No feasible initialization")
        self.elite=self._elite(initial)
        if family=="LMOCSO":self.pop=self._lm_select(initial)
        self._details()

    def _elite(self,pop):
        pop=unique_pop(pop,self.farm.n)
        front=pop.take(A.fronts(pop)[0])
        return A.gde_select(front,min(100,len(front)))

    def _lm_select(self,pop):
        # Offspring are supplied first: a neutral physical move keeps its new velocity.
        pop=unique_pop(pop,self.farm.n)
        primary=A.lm_select(pop,self.refs,self.theta)
        if len(primary)>self.nominal:primary=A.gde_select(primary,self.nominal)
        target=min(len(pop),max(min(self.active_floor,self.nominal),len(primary)))
        if len(primary)>=target:return primary
        occupied=set(layout_tokens(primary,self.farm.n))
        rest=pop.take([i for i,k in enumerate(layout_tokens(pop,self.farm.n)) if k not in occupied])
        return A.join(primary,A.gde_select(rest,target-len(primary)))

    def _evaluate(self,rows,anchors,velocity=None):
        rows=np.atleast_2d(rows)[:self.remaining]
        anchors=np.atleast_2d(anchors)[:len(rows)]
        v=np.zeros_like(rows) if velocity is None else velocity[:len(rows)].copy()
        xs=[];fs=[];cvs=[]
        for i,(row,anchor) in enumerate(zip(rows,anchors)):
            ids=selected(anchor,self.farm.n)
            xx,ff,cv=self.farm.evaluate_operator(row,ids)
            moved=np.abs(xx-row)>1e-12
            self.operator_counts["velocity_coordinates_reset"]+=int(np.count_nonzero(moved&(v[i]!=0)))
            v[i,moved]=0.
            xs.append(xx);fs.append(ff);cvs.append(cv)
        self.operator_counts["native_operator_requests"]+=len(rows)
        return A.Pop(np.array(xs),np.array(fs),np.array(cvs),v)

    def _lm_step(self):
        p=self.pop;fitness=A.sde(p.f)
        paired=self.rng.permutation(len(p))[:len(p)//2*2] if len(p)>=2 else np.array([0,0])
        loser,winner=np.split(paired,2)
        swap=fitness[loser]>=fitness[winner]
        loser[swap],winner[swap]=winner[swap].copy(),loser[swap].copy()
        r1,r2=self.rng.random((len(loser),1)),self.rng.random((len(loser),1))
        x,v=A.swarm_step(p.x[loser],p.v[loser],p.x[winner],r1,r2)
        rows=A.pm(np.vstack([x,p.x[winner]]),self.rng)[:self.remaining]
        anchors=np.vstack([p.x[loser],p.x[winner]])[:len(rows)]
        velocity=np.vstack([v,p.v[winner]])[:len(rows)]
        off=self._evaluate(rows,anchors,velocity)
        self.pop=self._lm_select(A.join(off,p))
        self.elite=self._elite(A.join(self.elite,off))
        self.operator_counts["competitive_swarm_steps"]+=1

    def _dgea_step(self):
        if self.off is None:
            rows=self.rng.random((min(self.nominal,self.remaining),self.farm.m))
            anchors=self.pop.x[np.arange(len(rows))%len(self.pop)]
            self.operator_counts["direction_initialization_batches"]+=1
        else:
            pool=unique_pop(A.join(self.off,self.pop),self.farm.n)
            # The original selected set and its rank labels remain authoritative.
            self.pop,ranks=A.dgea_pre(pool,self.refs,self.theta)
            rows,anchors=V6Problem._direction_rows(self,self.pop,ranks)
            self.operator_counts["direction_guided_steps"]+=1
        self.off=self._evaluate(rows,anchors)
        pool=unique_pop(A.join(self.off,self.archive),self.farm.n)
        self.archive=pool.take(A.apd_ids(pool,self.refs,self.theta))
        self.elite=self._elite(A.join(self.elite,self.off))

    def _local_proposal(self,parent,mode):
        """Cheap physical screening; every accepted proposal gets full exact objectives."""
        farm=self.farm;ids=selected(parent.x[0],farm.n)
        coupling=farm.ensure_influence()
        impact=coupling[:,ids].sum(axis=1)+coupling[ids,:].sum(axis=0)
        conflicts=farm.conflict[:,ids].sum(axis=1)
        membership=np.zeros(farm.m,bool);membership[ids]=True
        points=farm.xy[ids]
        hull=ConvexHull(points)
        inside=np.all(farm.xy@hull.equations[:,:2].T+hull.equations[:,2]<=1e-8,axis=1)
        if mode=="area":
            vertices=hull.vertices
            previous=points[np.roll(vertices,1)];current=points[vertices];following=points[np.roll(vertices,-1)]
            a=current-previous;b=following-previous
            score=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
            outgoing=ids[vertices[np.argsort(-score,kind="stable")[:6]]]
        elif mode=="cost":
            farm.ensure_ground()
            local=farm.ground_dist[np.ix_(ids,np.r_[ids,farm.m])].copy()
            local[np.arange(len(ids)),np.arange(len(ids))]=np.inf
            outgoing=ids[np.argsort(-local.min(axis=1),kind="stable")[:6]]
        else:
            outgoing=ids[np.argsort(-impact[ids],kind="stable")[:6]]
        proposals=[]
        for removed in outgoing:
            eligible=(conflicts-farm.conflict[:,removed].astype(int)==0)&~membership
            if mode=="area":eligible&=inside
            incoming=np.flatnonzero(eligible)
            if not len(incoming):continue
            retained=ids[ids!=removed]
            incoming_impact=impact[incoming]-coupling[incoming,removed]-coupling[removed,incoming]
            if mode=="cost":
                near=farm.ground_dist[np.ix_(incoming,np.r_[retained,farm.m])].min(axis=1)
                # Cost uses a cheap connection heuristic, not an uncounted objective evaluation.
                score=(near-near.min())/max(float(np.ptp(near)),1.)
                score+=.2*(incoming_impact-incoming_impact.min())/max(float(np.ptp(incoming_impact)),1e-12)
            else:score=incoming_impact-impact[removed]
            for index in np.argsort(score,kind="stable")[:8]:
                candidate=np.sort(np.r_[retained,incoming[index]])
                self.operator_counts["local_screened_candidates"]+=1
                if token(candidate) not in farm.cache:
                    proposals.append((float(score[index]),candidate))
                    break
        if not proposals:return None
        proposals.sort(key=lambda item:item[0])
        return proposals[0][1]

    def _refine(self,count):
        for _ in range(min(count,self.remaining)):
            mode=("energy","area","cost")[self.local_mode%3];self.local_mode+=1
            axis={"energy":0,"cost":1,"area":2}[mode]
            candidates=np.argsort(self.elite.f[:,axis],kind="stable")[:min(3,len(self.elite))]
            parent=self.elite.take([candidates[self.rng.integers(len(candidates))]])
            started=time.perf_counter()
            ids=self._local_proposal(parent,mode)
            self.farm.components["physical_search_screening_wall_seconds"]+=time.perf_counter()-started
            if ids is None:
                self.operator_counts["local_proposal_exhausted"]+=1
                continue
            before=selected(parent.x[0],self.farm.n)
            x,f,cv=self.farm.evaluate_feasible(ids,parent.x[0],before)
            child=A.Pop(x[None,:],f[None,:],np.array([cv]),np.zeros((1,self.farm.m)))
            self.operator_counts["local_exact_requests"]+=1
            self.operator_counts["local_"+mode+"_requests"]+=1
            self.operator_counts["local_"+mode+"_target_improvements"]+=int(f[axis]<parent.f[0,axis])
            self.elite=self._elite(A.join(child,self.elite))
            if self.family=="LMOCSO":self.pop=self._lm_select(A.join(child,self.pop))
            else:
                self.off=A.join(self.off,child)
                pool=unique_pop(A.join(child,self.archive),self.farm.n)
                self.archive=pool.take(A.apd_ids(pool,self.refs,self.theta))

    def step(self):
        if not self.remaining:return
        before=len(self.farm.records)
        if self.family=="LMOCSO":self._lm_step()
        else:self._dgea_step()
        used=len(self.farm.records)-before
        self._refine(max(1,used//4))
        self.record()

    def record(self):
        super().record()
        self.trace[-1]["operator_counts"]=dict(self.operator_counts)


def make_engine(farm,method,seed,budget=10000,population=100,**options):
    if method not in ("LMOCSO","DGEA","LMOCSO_problem","DGEA_problem"):
        raise ValueError("V7 only changes LMOCSO and DGEA")
    if options and options!={"layout_diversity":True,"compound":True,"max_attempts":24}:
        raise ValueError("This revision has one fixed integrated design; no parameter sweep")
    family=method.removesuffix("_problem")
    cls=Optimized if method.endswith("_problem") else Baseline
    return cls(farm,family,seed,budget,population)
