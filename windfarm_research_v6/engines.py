"""Family-preserving extensions: cooperative DE, competitive swarm, directions.

Frozen baselines use the original Engine unchanged. No global hooks are patched.
The extensions retain latent operator coordinates; feasibility, identity and
diversity are defined on the actual N selected candidate indices.
"""
from collections import Counter
import time
import numpy as np

from .compat import legacy as A, BASELINES, METHODS
from .feasible import selected, token, project


def layout_tokens(pop, n):
    return [token(selected(row, n)) for row in pop.x]


def diverse_select(pop, target, n, refs, theta, family, layout_diversity=True):
    """Pareto ranks first, family selection and physical diversity at truncation."""
    unique = {}
    for i, key in enumerate(layout_tokens(pop, n)):
        if key not in unique or pop.cv[i] < pop.cv[unique[key]]:
            unique[key] = i
    pop = pop.take(list(unique.values()))
    target = min(target, len(pop))
    if len(pop) <= target:
        return pop
    fronts = A.fronts(pop)
    spans = np.ptp(pop.f, axis=0)
    normalized = (pop.f-pop.f.min(axis=0))/np.maximum(spans, 1e-12)
    scaled = A.Pop(pop.x, normalized, pop.cv, pop.v)
    if family == "CCGDE3":
        primary_tokens = set(layout_tokens(A.gde_select(pop, target), n))
        primary = np.array([i for i,k in enumerate(layout_tokens(pop,n)) if k in primary_tokens])
    else:
        source = fronts[0] if family == "LMOCSO" else np.arange(len(pop))
        primary = source[A.apd_ids(scaled.take(source), refs, theta)]
    primary = set(map(int, primary))
    obj_distance = np.linalg.norm(normalized[:, None]-normalized[None, :], axis=2)/np.sqrt(pop.f.shape[1])
    if layout_diversity:
        membership = np.zeros((len(pop), pop.x.shape[1]), np.float32)
        for i, row in enumerate(pop.x):
            membership[i, selected(row,n)] = 1
        distance = .5*obj_distance+.5*np.maximum(0., 1-membership@membership.T/n)
    else:
        distance = obj_distance
    chosen = []
    for front in fronts:
        ids = list(map(int, front))
        if len(chosen)+len(ids) <= target:
            chosen.extend(ids)
            continue
        # Preserve objective extrema in the truncated nondominated front.
        if not chosen:
            for axis in range(pop.f.shape[1]):
                i = ids[int(np.argmin(pop.f[ids,axis]))]
                if i not in chosen and len(chosen) < target:
                    chosen.append(i)
        while len(chosen) < target:
            remaining = [i for i in ids if i not in chosen]
            preferred = [i for i in remaining if i in primary]
            pool = preferred if preferred else remaining
            if not pool:
                break
            values = distance[np.ix_(pool,chosen)].min(axis=1) if chosen else np.ones(len(pool))
            chosen.append(pool[int(np.argmax(values))])
        break
    return pop.take(chosen)


class BaselineEngine(A.Engine):
    def __init__(self, farm, family, seed, budget=10000, population=100):
        if family not in BASELINES:
            raise ValueError(f"Unknown baseline: {family}")
        if population < 4 or budget < population:
            raise ValueError("Require population>=4 and budget>=population")
        self.family = family
        self.display_method = family
        super().__init__(farm, BASELINES[family], seed, budget, population)

    def record(self):
        super().record()
        self._details()

    def _details(self):
        pop = self.archive if self.family == "DGEA" and hasattr(self,"archive") else self.pop
        self.trace[-1].update(population=len(pop), physical_unique=len(set(layout_tokens(pop,self.farm.n))),
                             family=self.family, method=self.display_method)


class ProblemEngine(BaselineEngine):
    def __init__(self, farm, family, seed, budget=10000, population=100,
                 layout_diversity=True, compound=True, max_attempts=24):
        self.operator_counts = Counter()
        self.layout_diversity = bool(layout_diversity)
        self.compound = bool(compound)
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 2:
            raise ValueError("At least two bounded projection attempts required")
        super().__init__(farm, family, seed, budget, population)
        self.display_method = family+"_problem"
        # Recover already-evaluated common initialization; no uncharged reevaluation.
        initial = A.Pop(np.array(farm.initial_keys[:population]),
                        np.array([np.array([-r[1][0],r[1][1],r[1][2]])/farm.scales if r[2]==0 else np.full(3,1e6) for r in farm.records[:population]]),
                        np.array([r[2] for r in farm.records[:population]]),
                        np.zeros((population,farm.m)))
        initial = initial.take(np.flatnonzero(initial.cv == 0))
        if not len(initial):
            raise ValueError("No feasible initial layout; do not infer global infeasibility")
        initial = self._select(initial,population)
        attempts = 0
        while len(initial) < population and self.remaining and attempts < 10*population:
            attempts += 1
            candidate = A.evaluate(farm,self.rng.random((1,farm.m)))
            if candidate.cv[0] == 0:
                initial = self._select(A.join(initial,candidate),population)
        self.operator_counts["initial_refill_requests"] = attempts
        self.pop = initial
        if family == "DGEA":
            self.archive = initial
            self.off = None
        elif family == "CCGDE3":
            self.context_archive = initial
            # Keep the baseline subgroup count and nominal species sizes.
            self.species = [self._select(A.join(p,initial),len(p)) for p in self.species]
            self.pop = self.species[0]
        self.trace[-1]["evaluations"] = len(farm.records)
        self._details()

    def _select(self,pop,target):
        return diverse_select(pop,target,self.farm.n,self.refs,self.theta,self.family,self.layout_diversity)

    def _guided(self, rows, anchors, velocity=None, group=None, require_change=True, avoid=()):
        rows = np.atleast_2d(rows)[:self.remaining]
        anchors = np.atleast_2d(anchors)[:len(rows)]
        v = np.zeros_like(rows) if velocity is None else velocity[:len(rows)].copy()
        returned, objectives, cv = [], [], []
        avoid = set(avoid)
        for i,(row,anchor_keys) in enumerate(zip(rows,anchors)):
            parent = selected(anchor_keys,self.farm.n)
            started = time.perf_counter()
            proposal = project(row,parent,self.farm.conflict,self.rng,group=group,
                               avoid=avoid,max_changes=(8 if self.compound else 1) if require_change else self.farm.n,
                               max_attempts=self.max_attempts,require_change=require_change)
            self.farm.components["problem_projection_wall_seconds"] += time.perf_counter()-started
            role = "offspring" if require_change else "context_parent"
            self.operator_counts[role+"_proposals"] += 1
            self.operator_counts[role+"_projection_attempts"] += proposal.attempts
            self.operator_counts[role+"_duplicate_rejections_before_evaluation"] += proposal.duplicate_rejections
            self.operator_counts[role+"_primary_duplicate_proposals"] += int(proposal.primary_duplicate)
            self.operator_counts[role+"_compound_changes"] += int(proposal.compound)
            self.operator_counts[role+"_unchanged_outputs"] += int(proposal.changed == 0)
            xx,ff,cc = self.farm.evaluate_layout(proposal.indices,row,parent=parent)
            v[i,np.abs(xx-row)>1e-12] = 0.
            returned.append(xx);objectives.append(ff);cv.append(cc)
            if require_change:
                avoid.add(token(proposal.indices))
        return A.Pop(np.array(returned),np.array(objectives),np.array(cv),v)

    def step(self):
        if not self.remaining:
            return
        if self.family == "CCGDE3":
            self._cc_step()
        elif self.family == "LMOCSO":
            self._lm_step()
        else:
            self._dgea_step()
        self.record()

    def _lm_step(self):
        p = self.pop
        fitness = A.sde(p.f)
        paired = self.rng.permutation(len(p))[:len(p)//2*2] if len(p)>=2 else np.array([0,0])
        loser,winner = np.split(paired,2)
        change = fitness[loser] >= fitness[winner]
        loser[change],winner[change] = winner[change].copy(),loser[change].copy()
        r1,r2 = self.rng.random((len(loser),1)),self.rng.random((len(loser),1))
        x,v = A.swarm_step(p.x[loser],p.v[loser],p.x[winner],r1,r2)
        rows = A.pm(np.vstack([x,p.x[winner]]),self.rng)[:self.remaining]
        velocity = np.vstack([v,p.v[winner]])[:len(rows)]
        anchors = np.vstack([p.x[loser],p.x[winner]])[:len(rows)]
        self.operator_counts["competitive_swarm_steps"] += 1
        off = self._guided(rows,anchors,velocity,avoid=layout_tokens(p,self.farm.n))
        self.pop = self._select(A.join(p,off),self.nominal)

    def _direction_rows(self,pop,ranks):
        # Same DGEA direction/distribution formula, additionally return the true anchor.
        nd,dom = pop.x[ranks==1],pop.x[ranks>1]
        start = nd[self.rng.integers(len(nd))]
        ref_no = 10
        if len(dom)<ref_no:
            take = min(len(nd),ref_no-len(dom))
            end = np.vstack([dom,nd[self.rng.permutation(len(nd))[:take]]])
        else:
            end = dom[self.rng.permutation(len(dom))[:ref_no]]
        directions = end-start
        directions /= np.maximum(np.linalg.norm(directions,axis=1)[:,None],1e-30)
        rows = []
        for direction in directions:
            lam = (nd-start)@direction
            sigma = (np.std(lam,ddof=1) if len(lam)>1 else 0.)*(1+(len(dom)+1)/len(pop))
            rows.append(start+self.rng.normal(0,sigma,(max(self.nominal//ref_no,10),1))*direction)
        rows = A.pm(np.clip(np.vstack(rows),0,1),self.rng)[:self.remaining]
        return rows,np.tile(start,(len(rows),1))

    def _dgea_step(self):
        if self.off is None:
            rows = self.rng.random((min(self.nominal,self.remaining),self.farm.m))
            anchors = self.pop.x[self.rng.integers(len(self.pop),size=len(rows))]
            self.operator_counts["direction_initial_offspring_batches"] += 1
        else:
            pool = A.join(self.pop,self.off)
            primary,_ = A.dgea_pre(pool,self.refs,self.theta)
            self.pop = self._select(A.join(primary,pool),self.nominal)
            ranks = np.empty(len(self.pop),int)
            for rank,ids in enumerate(A.fronts(self.pop),1):
                ranks[ids] = rank
            rows,anchors = self._direction_rows(self.pop,ranks)
            self.operator_counts["direction_guided_steps"] += 1
        self.off = self._guided(rows,anchors,avoid=layout_tokens(self.pop,self.farm.n))
        self.archive = self._select(A.join(self.archive,self.off),self.nominal)

    def _cross_group(self,p):
        parent = p.take([self.rng.integers(len(p))])
        rows = A.pm(parent.x,self.rng)
        self.operator_counts["cross_group_exchange_requests"] += 1
        off = self._guided(rows,parent.x,avoid=layout_tokens(p,self.farm.n))
        return self._select(A.join(p,off),len(p)),off

    def _cc_step(self):
        k = self.group_index
        group = self.group_list[k]
        p = self.species[k]
        count = min(len(p),self.remaining//2)
        if count == 0 or len(p)<3:
            self.species[k],off = self._cross_group(p)
            self.context_archive = self._select(A.join(self.context_archive,off),self.nominal)
            self.pop = self.species[k]
            return
        fronts = A.fronts(self.context_archive)
        context = self.context_archive.x[self.rng.choice(fronts[0])].copy()
        anchors = np.tile(context,(count,1))
        rows = anchors.copy()
        rows[:,group] = p.x[:count,group]
        parents = self._guided(rows,anchors,group=group,require_change=False)
        donor = np.empty((count,len(group)))
        for i in range(count):
            candidates = np.delete(np.arange(len(p)),i)
            a,b = self.rng.choice(candidates,2,replace=False)
            donor[i] = p.x[i,group]+.5*(p.x[a,group]-p.x[b,group])
        mask = self.rng.random(donor.shape)<.5
        children = parents.x.copy()
        children[:,group] = A.pm(np.where(mask,donor,p.x[:count,group]),self.rng)
        self.operator_counts["cooperative_differential_steps"] += 1
        child = self._guided(children,parents.x,group=group,avoid=layout_tokens(p,self.farm.n))
        retained = []
        for i in range(count):
            if np.all(child.f[i]<=parents.f[i]):
                retained.append(count+i)
            elif np.all(parents.f[i]<=child.f[i]):
                retained.append(i)
            else:
                retained.extend([i,count+i])
        pool = A.join(parents,child).take(retained)
        # Full-layout parent candidates preserve enough distinct survivors.
        self.species[k] = self._select(A.join(pool,p),len(p))
        self.context_archive = self._select(A.join(self.context_archive,parents,child),self.nominal)
        self.group_index = (k+1)%len(self.group_list)
        if self.group_index == 0 and self.remaining:
            self.species[k],off = self._cross_group(self.species[k])
            self.context_archive = self._select(A.join(self.context_archive,off),self.nominal)
        self.pop = self.species[k]

    def record(self):
        super().record()
        self.trace[-1]["operator_counts"] = dict(self.operator_counts)


def make_engine(farm,method,seed,budget=10000,population=100,**options):
    if method not in METHODS:
        raise ValueError(f"Choose one of {METHODS}")
    if method.endswith("_problem"):
        return ProblemEngine(farm,method[:-8],seed,budget,population,**options)
    if options:
        raise ValueError("Problem-specific switches cannot silently alter a baseline")
    return BaselineEngine(farm,method,seed,budget,population)
