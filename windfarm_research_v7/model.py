"""Exact dense MST acceleration and sparse state writeback; same objectives."""
import math
import time
import numpy as np
from shapely.geometry import MultiPoint

from windfarm_research_v6.model import OverlapFarm
from windfarm_research_v6.compat import repair_keys, physical_power
from windfarm_research_v6.feasible import selected, token


def dense_prim_length(matrix):
    """Exact MST on the supplied complete weighted graph (no k-NN pruning)."""
    weights=np.asarray(matrix,float)
    if weights.ndim!=2 or weights.shape[0]!=weights.shape[1]:
        raise ValueError("Square distance matrix required")
    n=len(weights)
    if n<2:return 0.
    used=np.zeros(n,bool);used[0]=True
    nearest=weights[0].copy();nearest[0]=np.inf
    lengths=[]
    for _ in range(n-1):
        vertex=int(np.argmin(nearest))
        if not np.isfinite(nearest[vertex]):raise ValueError("Disconnected distance graph")
        lengths.append(float(nearest[vertex]));used[vertex]=True
        np.minimum(nearest,weights[vertex],out=nearest)
        nearest[used]=np.inf
    return math.fsum(lengths)


def sparse_writeback(scores,indices):
    """Change only membership-flipped coordinates, plus unavoidable boundary ties."""
    original=np.clip(np.asarray(scores,float),0.,1.)
    indices=np.sort(np.asarray(indices,int))
    if original.ndim!=1 or not np.isfinite(original).all():raise ValueError("Invalid operator coordinates")
    n=len(indices);raw=selected(original,n)
    if np.array_equal(raw,indices):return original.copy()
    result=original.copy()
    desired=np.zeros(len(original),bool);desired[indices]=True
    was=np.zeros(len(original),bool);was[raw]=True
    added=desired&~was;removed=was&~desired
    fixed_selected=desired&was;fixed_unselected=~desired&~was
    low=float(original[fixed_selected].max()) if fixed_selected.any() else 0.
    high=float(original[fixed_unselected].min()) if fixed_unselected.any() else 1.
    if low<high:
        a=(2*low+high)/3;b=(low+2*high)/3
        if not a<b:a,b=low,high
        result[added]=a;result[removed]=b
    elif low==0.:
        result[added]=0.
        result[~desired&(result==0.)]=np.finfo(float).eps
        result[removed]=max(np.finfo(float).eps,high)
    else:
        below=np.nextafter(low,0.)
        result[added]=below
        result[desired&(result>=low)]=below
        result[removed]=low
    if not np.array_equal(selected(result,n),indices):
        raise ArithmeticError("Sparse writeback failed to preserve the chosen layout")
    return result


class FastFarm(OverlapFarm):
    def __init__(self,*args,**kwargs):
        self.clock_wall=time.perf_counter();self.clock_cpu=time.process_time()
        self.clock_offsets=(0.,0.)
        self.influence_kernel=None
        super().__init__(*args,**kwargs)

    def __getstate__(self):
        state=super().__getstate__()
        state["influence_kernel"]=None
        state["clock_offsets"]=self.elapsed()
        return state

    def __setstate__(self,state):
        wall,cpu=time.perf_counter(),time.process_time()
        super().__setstate__(state)
        self.clock_wall,self.clock_cpu=wall,cpu

    def elapsed(self):
        return (self.clock_offsets[0]+time.perf_counter()-self.clock_wall,
                self.clock_offsets[1]+time.process_time()-self.clock_cpu)

    def _wake_sums(self,indices,parent=None,force_full=False):
        if parent is not None and token(parent) in self.sum_cache:
            self.sum_cache.move_to_end(token(parent))
        return super()._wake_sums(indices,parent,force_full)

    def ensure_influence(self):
        if self.influence_kernel is None:
            started=time.perf_counter();self.ensure_wake()
            weights=self.scene.omega/self.scene.omega.sum()
            self.influence_kernel=np.tensordot(weights,self.wake,axes=(0,0))
            self.components["search_influence_setup_wall_seconds"]+=time.perf_counter()-started
        return self.influence_kernel

    def physical_metrics(self,indices,parent=None,force_full=False):
        indices=np.asarray(indices)
        if indices.dtype.kind not in "iu":raise ValueError("Integer candidate indices required")
        indices=np.sort(indices.astype(int))
        if not self.feasible(indices):raise ValueError("Infeasible layout")
        started=time.perf_counter()
        sums=self._wake_sums(indices,parent,force_full)
        energy=float(physical_power(self.scene,np.sqrt(sums)).sum()*8760/1e6)
        self.components["energy_wall_seconds"]+=time.perf_counter()-started
        started=time.perf_counter();self.ensure_ground()
        ids=np.r_[indices,self.m]
        unique=np.unique(self.ground_points[ids],axis=0,return_index=True)[1]
        ids=ids[unique]
        length=dense_prim_length(self.ground_dist[np.ix_(ids,ids)])
        cost=self.economics.compute(self.scene.rated_kw,self.n,length,self.reference_length,energy)
        self.components["cost_and_connection_wall_seconds"]+=time.perf_counter()-started
        self.statistics["exact_dense_MST_evaluations"]+=1
        started=time.perf_counter();hull=MultiPoint(self.xy[indices]).convex_hull
        gross,area=hull.area/1e6,hull.intersection(self.allowed).area/1e6
        self.components["area_wall_seconds"]+=time.perf_counter()-started
        metrics=np.array([energy,cost["lcoe"],area,length,gross,gross-area,
                          cost["capex"],cost["annual_cost"],cost["electrical_capex"],self.reference_length])
        if not np.isfinite(metrics).all() or energy<=0:raise ValueError("Invalid metrics")
        return metrics

    def evaluate_operator(self,scores,parent=None):
        original=np.clip(np.asarray(scores,float),0.,1.)
        if original.shape!=(self.m,) or not np.isfinite(original).all():raise ValueError("Invalid score vector")
        started=time.perf_counter();raw=selected(original,self.n)
        pairs=int(self.conflict[np.ix_(raw,raw)].sum()//2)
        if pairs==0:
            ids=raw;ok=True
            self.statistics["native_feasible_proposals"]+=1
        else:
            _,ids,_,ok=repair_keys(original,self.conflict,self.n)
            if not ok and parent is not None:
                ids=np.sort(np.asarray(parent,int));ok=self.feasible(ids)
                self.statistics["repair_failure_parent_fallbacks"]+=1
        written=sparse_writeback(original,ids) if ok else original.copy()
        self.statistics["writeback_coordinates"]+=int(np.count_nonzero(written!=original))
        if parent is not None:
            changed=self.n-len(np.intersect1d(ids,parent))
            self.statistics["accepted_changes_over_eight"]+=int(ok and changed>8)
            self.statistics["neutral_layout_proposals"]+=int(changed==0)
        self.components["decode_wall_seconds"]+=time.perf_counter()-started
        return self._record(original,written,ids,pairs,ok,parent)

    def evaluate_feasible(self,ids,scores,parent):
        ids=np.sort(np.asarray(ids,int))
        if not self.feasible(ids):raise ValueError("Local refinement violated hard constraints")
        original=np.clip(np.asarray(scores,float),0.,1.)
        written=sparse_writeback(original,ids)
        return self._record(original,written,ids,0,True,parent)

    def case_metadata(self):
        result=super().case_metadata()
        result.update(evaluation_backend="v7 exact dense Prim; same complete graph and cost function",
                      search_surrogate="Frequency-weighted squared wake kernel guides proposals only; exact objectives determine selection")
        return result
