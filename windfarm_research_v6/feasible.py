"""Operator-guided feasible projection with an explicit fixed outside context."""
from dataclasses import dataclass
import numpy as np


def selected(keys, n):
    return np.sort(np.argsort(keys, kind="stable")[:n])


def token(indices):
    return np.sort(np.asarray(indices, dtype="<u2")).tobytes()


def validate(indices, conflict, n):
    ids = np.asarray(indices)
    if ids.shape != (n,) or ids.dtype.kind not in "iu":
        return False
    return bool(len(np.unique(ids)) == n and np.all(ids >= 0) and
                np.all(ids < len(conflict)) and not conflict[np.ix_(ids, ids)].any())


def encode(scores, indices):
    scores = np.clip(np.asarray(scores, float), 0., 1.)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Finite score vector required")
    order = np.argsort(scores, kind="stable")
    keep = np.zeros(len(scores), bool)
    keep[indices] = True
    values = np.sort(scores)
    if np.any(np.diff(values) <= 1e-12):
        values = (np.arange(len(scores))+.5)/len(scores)
    result = np.empty_like(scores)
    result[np.r_[order[keep[order]], order[~keep[order]]]] = values
    return result


@dataclass
class Proposal:
    indices: np.ndarray
    keys: np.ndarray
    attempts: int
    duplicate_rejections: int
    primary_duplicate: bool
    compound: bool
    changed: int
    fallback: bool


def project(scores, anchor, conflict, rng, group=None, avoid=(), max_changes=8,
            max_attempts=24, require_change=True):
    scores = np.clip(np.asarray(scores, float), 0., 1.)
    anchor = np.sort(np.asarray(anchor, int))
    n, m = len(anchor), len(conflict)
    if scores.shape != (m,) or not np.isfinite(scores).all() or not validate(anchor, conflict, n):
        raise ValueError("Invalid proposal or feasible anchor")
    group = np.arange(m) if group is None else np.unique(np.asarray(group, int))
    if np.any(group < 0) or np.any(group >= m):
        raise ValueError("Invalid group")
    outside = np.setdiff1d(anchor, group)
    local = np.intersect1d(anchor, group)
    avoid = set(avoid)
    parent_token = token(anchor)
    order = group[np.argsort(scores[group], kind="stable")]
    attempts, duplicates, primary_duplicate = 1, 0, False

    def fill(prefix, target):
        chosen = list(map(int, prefix))
        blocked = conflict[chosen].any(axis=0) if chosen else np.zeros(m, bool)
        blocked[chosen] = True
        for index in order:
            if len(chosen) >= target:
                break
            if not blocked[index]:
                chosen.append(int(index))
                blocked |= conflict[index]
                blocked[index] = True
        return np.sort(np.array(chosen, int))

    def admissible(ids):
        nonlocal duplicates
        if not validate(ids, conflict, n):
            return False
        # Also prevents accidental loss of an outside-context point.
        if not np.array_equal(np.setdiff1d(ids, group), outside):
            raise AssertionError("Outside-group decision changed")
        changed = n-len(np.intersect1d(ids, anchor))
        key = token(ids)
        if key in avoid or (require_change and key == parent_token):
            duplicates += 1
            return False
        return changed <= max_changes

    first = fill(outside, n)
    if len(first) == n:
        primary_duplicate = token(first) in avoid or token(first) == parent_token
    if admissible(first):
        changed = n-len(np.intersect1d(first, anchor))
        return Proposal(first, encode(scores, first), attempts, duplicates,
                        primary_duplicate, changed > 1, changed, False)

    candidates = np.setdiff1d(group, anchor)
    candidates = candidates[np.argsort(scores[candidates], kind="stable")]
    guided = candidates[:max_attempts//2]
    random_part = rng.permutation(candidates[max_attempts//2:])
    choices = np.r_[guided, random_part][:max_attempts-1]
    for incoming in choices:
        attempts += 1
        if len(outside) and conflict[incoming, outside].any():
            continue
        remove = local[conflict[incoming, local]]
        if not len(remove):
            if not len(local):
                continue
            remove = np.array([local[np.argmax(scores[local])]])
        if len(remove) > max_changes:
            continue
        prefix = np.r_[np.setdiff1d(anchor, remove), incoming]
        trial = fill(prefix, n)
        if admissible(trial):
            changed = n-len(np.intersect1d(trial, anchor))
            return Proposal(trial, encode(scores, trial), attempts, duplicates,
                            primary_duplicate, changed > 1, changed, True)
    # Bounded failure: retain the true parent and expose the failure in telemetry.
    # This consumes an evaluation request (normally a cache hit), so it cannot hang.
    return Proposal(anchor.copy(), encode(scores, anchor), attempts, duplicates,
                    primary_duplicate, False, 0, True)
