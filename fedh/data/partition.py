"""Entity-ownership partitions with a controlled cross-client ratio rho.

Start from a community-based assignment (few cross-client paths), then shuffle
the owner labels of a growing fraction f of entities (more cross-client paths).
Shuffling labels among the chosen entities keeps every client's size exactly
fixed. f is found by bisection so that the achieved rho on a calibration set
of questions matches the target.
"""
from __future__ import annotations

import numpy as np

from .graph import RelGraph, follow_path


def target_sizes(n_ent: int, n_clients: int, balance: str, alpha: float, rng) -> np.ndarray:
    if n_clients == 1:
        return np.array([n_ent])
    if balance == "balanced":
        w = np.ones(n_clients) / n_clients
    else:
        w = rng.dirichlet(alpha * np.ones(n_clients))
        w = np.maximum(w, 0.02)          # no client below 2% of entities
        w = w / w.sum()
    sizes = np.floor(w * n_ent).astype(int)
    sizes[: n_ent - sizes.sum()] += 1
    return sizes


def community_assignment(kb_heads, kb_tails, n_ent, sizes, seed: int) -> np.ndarray:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_ent))
    G.add_edges_from(zip(kb_heads.tolist(), kb_tails.tolist()))
    try:
        comms = nx.community.louvain_communities(G, seed=seed)
    except AttributeError:  # networkx < 2.8
        comms = list(nx.algorithms.community.label_propagation_communities(G))
    chunks = sorted((sorted(c) for c in comms), key=len, reverse=True)
    remaining = sizes.astype(int).copy()
    owner = np.full(n_ent, -1, dtype=np.int64)
    queue = list(chunks)
    while queue:
        c = queue.pop(0)
        b = int(np.argmax(remaining))
        take = min(len(c), int(remaining[b]))
        if take == 0:
            break
        owner[c[:take]] = b
        remaining[b] -= take
        if take < len(c):
            queue.insert(0, c[take:])
    left = np.where(owner < 0)[0]
    if left.size:  # numerical leftovers
        fill = np.repeat(np.arange(len(sizes)), np.maximum(remaining, 0))[: left.size]
        owner[left[: fill.size]] = fill
        owner[owner < 0] = 0
    return owner


def bfs_assignment(kb_heads, kb_tails, n_ent, sizes, seed: int) -> np.ndarray:
    """Fast community-like partition for large graphs: grow one region per client by
    round-robin breadth-first search from random seeds, each up to its target size."""
    from collections import deque
    rng = np.random.default_rng(seed)
    h = np.concatenate([kb_heads, kb_tails]).astype(np.int64)
    t = np.concatenate([kb_tails, kb_heads]).astype(np.int64)
    order = np.argsort(h, kind="stable")
    h, t = h[order], t[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(h, minlength=n_ent))])
    owner = np.full(n_ent, -1, dtype=np.int64)
    remaining = sizes.astype(int).copy()
    n_cl = len(sizes)
    queues = [deque() for _ in range(n_cl)]
    perm = rng.permutation(n_ent)
    nxt_seed = 0

    def new_seed():
        nonlocal nxt_seed
        while nxt_seed < n_ent and owner[perm[nxt_seed]] >= 0:
            nxt_seed += 1
        return int(perm[nxt_seed]) if nxt_seed < n_ent else -1

    for c in range(n_cl):
        s0 = new_seed()
        if s0 >= 0:
            owner[s0] = c
            remaining[c] -= 1
            queues[c].append(s0)
    active = [c for c in range(n_cl) if remaining[c] > 0]
    while active:
        still = []
        for c in active:
            claimed = 0
            while queues[c] and claimed < 64 and remaining[c] > 0:
                u = queues[c].popleft()
                for v in t[indptr[u]:indptr[u + 1]].tolist():
                    if owner[v] < 0 and remaining[c] > 0:
                        owner[v] = c
                        remaining[c] -= 1
                        queues[c].append(v)
                        claimed += 1
            if remaining[c] > 0 and not queues[c]:
                s0 = new_seed()          # region exhausted its component: jump to a new seed
                if s0 >= 0:
                    owner[s0] = c
                    remaining[c] -= 1
                    queues[c].append(s0)
            if remaining[c] > 0 and (queues[c] or nxt_seed < n_ent):
                still.append(c)
        active = still
    left = np.where(owner < 0)[0]
    if left.size:
        fill = np.repeat(np.arange(n_cl), np.maximum(remaining, 0))[: left.size]
        owner[left[: fill.size]] = fill
        owner[owner < 0] = 0
    return owner


def mix_assignment(base: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Shuffle owner labels among the first `frac` share of a fixed random entity order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(base.size)
    k = int(round(frac * base.size))
    owner = base.copy()
    if k > 1:
        chosen = order[:k]
        owner[chosen] = base[chosen][np.random.default_rng(seed + 1).permutation(k)]
    return owner


def is_cross_client(g: RelGraph, q: dict, owner: np.ndarray, no_return: bool = False) -> bool:
    """A question is local if owner(e0) alone reaches every answer its gold path reaches."""
    if not q.get("gold_paths"):
        return False
    seq = q["gold_paths"][0]
    e0 = q["topic_id"]
    ans = set(q["answers"])
    full = set(int(x) for x in follow_path(g, e0, seq, no_return=no_return)) & ans
    loc = set(int(x) for x in follow_path(g, e0, seq, owner, int(owner[e0]), no_return=no_return)) & ans
    return loc != full


def cross_ratio(g, questions, owner, no_return: bool = False) -> float:
    qs = [q for q in questions if q.get("gold_paths")]
    if not qs:
        return 0.0
    return float(np.mean([is_cross_client(g, q, owner, no_return) for q in qs]))


def calibrate_rho(g, base, calib_questions, target: float, seed: int, iters: int = 14, tol: float = 0.01,
                  no_return: bool = False):
    """Bisection on the shuffled fraction f; returns (owner, f, achieved_rho, trace)."""
    trace = []
    lo, hi = 0.0, 1.0
    r_lo = cross_ratio(g, calib_questions, mix_assignment(base, lo, seed), no_return)
    r_hi = cross_ratio(g, calib_questions, mix_assignment(base, hi, seed), no_return)
    trace += [(lo, r_lo), (hi, r_hi)]
    if target <= r_lo:
        return mix_assignment(base, lo, seed), lo, r_lo, trace
    if target >= r_hi:
        return mix_assignment(base, hi, seed), hi, r_hi, trace
    best = (abs(r_lo - target), lo, r_lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        r = cross_ratio(g, calib_questions, mix_assignment(base, mid, seed), no_return)
        trace.append((mid, r))
        if abs(r - target) < best[0]:
            best = (abs(r - target), mid, r)
        if abs(r - target) <= tol:
            break
        if r < target:
            lo = mid
        else:
            hi = mid
    _, f, r = best
    return mix_assignment(base, f, seed), f, r, trace
