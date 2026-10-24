"""Source-guided constrained random-key adaptations; see reference/SOURCE.json.

LMOCSO/DGEA operators follow PlatEMO; this is not a MATLAB equivalence claim.
Constraint ordering, key repair/writeback, fixed objective units and numerical
zero guards are explicit course adaptations. CCGDE3 also uses >2 species.
"""
import bootstrap
from dataclasses import dataclass
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

METHODS = ['NSGA2', 'NSGA3', 'CCGDE3_random', 'CCGDE3_wake', 'LMOCSO', 'DGEA']
NDS = NonDominatedSorting()


@dataclass
class Pop:
    x: np.ndarray
    f: np.ndarray
    cv: np.ndarray
    v: np.ndarray

    def __len__(self):
        return len(self.x)

    def take(self, ids):
        ids = np.atleast_1d(ids)
        return Pop(self.x[ids].copy(), self.f[ids].copy(), self.cv[ids].copy(), self.v[ids].copy())


def join(*pops):
    return Pop(*(np.concatenate([getattr(p, k) for p in pops]) for k in ('x', 'f', 'cv', 'v')))


def evaluate(farm, x, v=None):
    x = np.atleast_2d(x)
    velocity = np.zeros_like(x) if v is None else v.copy()
    repaired, f, cv = [], [], []
    for i, row in enumerate(x):
        xx, ff, cc = farm.evaluate(row)
        # Reset only repair-displaced velocity coordinates, keeping valid
        # swarm momentum elsewhere. Mutation does not itself reset velocity.
        velocity[i, np.abs(xx - row) > 1e-12] = 0
        repaired.append(xx)
        f.append(ff)
        cv.append(cc)
    return Pop(np.array(repaired), np.array(f), np.array(cv), velocity)


def fronts(pop):
    feasible = np.flatnonzero(pop.cv <= 0)
    result = [feasible[f] for f in NDS.do(pop.f[feasible])] if len(feasible) else []
    invalid = np.flatnonzero(pop.cv > 0)
    for value in np.unique(pop.cv[invalid]):
        result.append(invalid[pop.cv[invalid] == value])
    return result


def crowding(f):
    n, m = f.shape
    result = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        ids = np.argsort(f[:, j], kind='stable')
        width = f[ids[-1], j] - f[ids[0], j]
        if width <= 1e-14:
            continue
        result[ids[[0, -1]]] = np.inf
        result[ids[1:-1]] += (f[ids[2:], j] - f[ids[:-2], j]) / width
    return result


def gde_select(pop, size):
    chosen = []
    for ids in fronts(pop):
        if len(chosen) + len(ids) <= size:
            chosen.extend(ids)
        else:
            ids = list(ids)
            while len(ids) + len(chosen) > size:
                # GDE3 iterative crowding deletion, not one-shot truncation.
                ids.pop(int(np.argmin(crowding(pop.f[ids]))))
            chosen.extend(ids)
            break
    return pop.take(chosen)


def pm(x, rng, eta=20):
    x = np.clip(x.copy(), 0, 1)
    site = rng.random(x.shape) < 1 / x.shape[1]
    mu = rng.random(x.shape)
    low, high = site & (mu <= .5), site & (mu > .5)
    x[low] += (2 * mu[low] + (1 - 2 * mu[low]) * (1 - x[low]) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    x[high] += 1 - (2 * (1 - mu[high]) + 2 * (mu[high] - .5) * x[high] ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(x, 0, 1)


def sde(f):
    span = np.ptp(f, axis=0)
    z = (f - f.min(axis=0)) / np.where(span > 1e-14, span, 1)
    dist = np.linalg.norm(np.maximum(z[None, :, :] - z[:, None, :], 0), axis=2)
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def swarm_step(lx, lv, wx, r1, r2):
    velocity = r1 * lv + r2 * (wx - lx)
    return lx + velocity + r1 * (velocity - lv), velocity


def apd_ids(pop, refs, theta, ideal=None):
    if not len(pop):
        return np.array([], int)
    z = pop.f - (pop.f.min(axis=0) if ideal is None else ideal)
    norms = np.linalg.norm(z, axis=1)
    unit = refs / np.linalg.norm(refs, axis=1)[:, None]
    cosine = np.clip(unit @ unit.T, -1, 1)
    np.fill_diagonal(cosine, 0)
    gamma = np.maximum(np.arccos(cosine).min(axis=1), 1e-12)
    angles = np.arccos(np.clip(z @ unit.T / np.maximum(norms[:, None], 1e-30), -1, 1))
    angles[norms <= 1e-14] = 0
    association = angles.argmin(axis=1)
    chosen = []
    for i in np.unique(association):
        ids = np.flatnonzero(association == i)
        valid = ids[pop.cv[ids] <= 0]
        if len(valid):
            score = (1 + pop.f.shape[1] * theta * angles[valid, i] / gamma[i]) * norms[valid]
            chosen.append(valid[np.argmin(score)])
        else:
            chosen.append(ids[np.argmin(pop.cv[ids])])
    return np.array(chosen, int)


def lm_select(pop, refs, theta):
    nd = pop.take(fronts(pop)[0])
    return nd.take(apd_ids(nd, refs, theta))


def dgea_pre(pop, refs, theta, ref_no=10):
    fs = fronts(pop)
    ranks = np.full(len(pop), np.inf)
    count, last_rank = 0, 0
    for i, ids in enumerate(fs):
        ranks[ids] = i + 1
        count += len(ids)
        last_rank = i + 1
        if count >= min(len(refs), len(pop)):
            break
    if len(fs[0]) >= ref_no:
        chosen = list(np.flatnonzero(ranks < last_rank))
        last = np.flatnonzero(ranks == last_rank)
    else:
        chosen = list(fs[0])
        last = np.flatnonzero(ranks > 1)
    if len(last):
        chosen.extend(last[apd_ids(pop.take(last), refs, theta, pop.f.min(axis=0))])
    chosen = np.array(sorted(set(chosen)), int)
    return pop.take(chosen), ranks[chosen]


def direction_children(pop, ranks, rng, nominal=100, ref_no=10):
    nd, dom = pop.x[ranks == 1], pop.x[ranks > 1]
    start = nd[rng.integers(len(nd))]
    if len(dom) < ref_no:
        take = min(len(nd), ref_no - len(dom))
        end = np.vstack([dom, nd[rng.permutation(len(nd))[:take]]])
    else:
        end = dom[rng.permutation(len(dom))[:ref_no]]
    directions = end - start
    length = np.linalg.norm(directions, axis=1)
    directions /= np.maximum(length[:, None], 1e-30)
    sub_n = max(nominal // ref_no, 10)
    children = []
    for direction in directions:
        lam = (nd - start) @ direction
        sigma = (np.std(lam, ddof=1) if len(lam) > 1 else 0.) * (1 + (len(dom) + 1) / len(pop))
        children.append(start + rng.normal(0, sigma, (sub_n, 1)) * direction)
    return pm(np.clip(np.vstack(children), 0, 1), rng)


class Engine:
    def __init__(self, farm, method, seed, budget=10000, population=100):
        self.farm, self.method, self.budget, self.nominal = farm, method, budget, population
        self.rng = np.random.default_rng(seed)
        # All methods start with exactly the same evaluated 100 keys per seed.
        initial = self.rng.random((population, farm.m))
        self.refs = get_reference_directions('das-dennis', 3, n_partitions=12)
        self.iteration = 0
        self.trace = []
        if method in ('NSGA2', 'NSGA3'):
            from pymoo.algorithms.moo.nsga2 import NSGA2
            from pymoo.algorithms.moo.nsga3 import NSGA3
            from pymoo.core.problem import Problem
            problem = Problem(n_var=farm.m, n_obj=3, n_ieq_constr=1, xl=0, xu=1)
            cls = NSGA2 if method == 'NSGA2' else NSGA3
            kw = {} if method == 'NSGA2' else {'ref_dirs': self.refs}
            self.algorithm = cls(pop_size=population, sampling=initial, eliminate_duplicates=False, **kw)
            self.algorithm.setup(problem, termination=('n_eval', budget), seed=seed, verbose=False)
            self.step()
        else:
            self.pop = evaluate(farm, initial)
            if method == 'LMOCSO':
                self.pop = lm_select(self.pop, self.refs, self.theta)
            elif method == 'DGEA':
                self.archive = self.pop
                self.off = None  # source uses another random population, charged at next step
            elif method.startswith('CCGDE3'):
                self.group_list = farm.groups(self.rng, 'wake' if method.endswith('wake') else 'random')
                self.species = [self.pop.take(self.rng.permutation(population)[:40]) for g in self.group_list]
                self.group_index = 0
            else:
                raise ValueError(method)
            self.record()

    @property
    def theta(self):
        return (len(self.farm.records) / self.budget) ** 2

    @property
    def remaining(self):
        return self.budget - len(self.farm.records)

    def record(self):
        self.iteration += 1
        self.trace.append({'evaluations': len(self.farm.records), 'iteration': self.iteration})

    def step(self):
        if self.remaining <= 0:
            return
        if self.method in ('NSGA2', 'NSGA3'):
            infill = self.algorithm.ask()
            if infill is None or len(infill) == 0:
                raise RuntimeError('Empty pymoo infill')
            infill = infill[:self.remaining]
            p = evaluate(self.farm, infill.get('X'))
            infill.set('X', p.x, 'F', p.f, 'G', p.cv[:, None], 'H', np.empty((len(p), 0)))
            for ind in infill:
                ind.evaluated.update(['F', 'G', 'H'])
            self.algorithm.evaluator.n_eval += len(p)
            self.algorithm.tell(infills=infill)
        elif self.method == 'LMOCSO':
            p = self.pop
            fitness = sde(p.f)
            paired = self.rng.permutation(len(p))[:len(p) // 2 * 2] if len(p) >= 2 else np.array([0, 0])
            loser, winner = np.split(paired, 2)
            change = fitness[loser] >= fitness[winner]
            loser[change], winner[change] = winner[change].copy(), loser[change].copy()
            # Constraint preference also applies to competition.
            reverse = (p.cv[loser] < p.cv[winner])
            loser[reverse], winner[reverse] = winner[reverse].copy(), loser[reverse].copy()
            r1, r2 = self.rng.random((len(loser), 1)), self.rng.random((len(loser), 1))
            x, v = swarm_step(p.x[loser], p.v[loser], p.x[winner], r1, r2)
            x = pm(np.vstack([x, p.x[winner]]), self.rng)[:self.remaining]
            v = np.vstack([v, p.v[winner]])[:len(x)]
            off = evaluate(self.farm, x, v)
            self.pop = lm_select(join(p, off), self.refs, self.theta)
        elif self.method == 'DGEA':
            if self.off is None:
                self.off = evaluate(self.farm, self.rng.random((min(self.nominal, self.remaining), self.farm.m)))
            else:
                self.pop, ranks = dgea_pre(join(self.pop, self.off), self.refs, self.theta)
                children = direction_children(self.pop, ranks, self.rng, self.nominal)
                self.off = evaluate(self.farm, children[:self.remaining])
                pool = join(self.archive, self.off)
                self.archive = pool.take(apd_ids(pool, self.refs, self.theta))
        else:
            self.cc_step()
        self.record()

    def cc_step(self):
        k = self.group_index
        group = self.group_list[k]
        p = self.species[k]
        n = min(len(p), self.remaining // 2)
        if n == 0:
            # An odd final request evaluates an existing solution and is logged.
            evaluate(self.farm, p.x[:1])
            return
        # Freeze a common collaboration vector for all comparisons this step.
        context = p.x[0].copy()
        for j, g in enumerate(self.group_list):
            rep = self.species[j].take(fronts(self.species[j])[0])
            context[g] = rep.x[self.rng.integers(len(rep)), g]
        parents = np.tile(context, (n, 1))
        parents[:, group] = p.x[:n, group]
        donor = np.empty((n, len(group)))
        for i in range(n):
            ids = np.delete(np.arange(len(p)), i)
            a, b = self.rng.choice(ids, 2, replace=False)
            donor[i] = p.x[i, group] + .5 * (p.x[a, group] - p.x[b, group])
        mask = self.rng.random(donor.shape) < .5
        component = np.where(mask, donor, p.x[:n, group])
        offspring = parents.copy()
        offspring[:, group] = pm(component, self.rng)
        parent = evaluate(self.farm, parents)
        child = evaluate(self.farm, offspring)
        retained = []
        both = join(parent, child)
        for i in range(n):
            pc, cc = parent.cv[i], child.cv[i]
            if cc < pc or (cc == pc == 0 and np.all(child.f[i] <= parent.f[i])):
                retained.append(n + i)
            elif pc < cc or (pc == cc == 0 and np.all(parent.f[i] <= child.f[i])):
                retained.append(i)
            elif pc == cc and pc > 0:
                retained.append(n + i)
            else:
                retained.extend([i, n + i])
        pool = both.take(retained)
        if n < len(p):
            pool = join(pool, p.take(np.arange(n, len(p))))
        self.species[k] = gde_select(pool, len(p))
        self.group_index = (k + 1) % len(self.group_list)
        self.pop = self.species[k]
