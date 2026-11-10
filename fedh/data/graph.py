"""Compact relation graph with set-based expansion and relation-path search."""
from __future__ import annotations


import numpy as np


class RelGraph:
    """CSR over head entities; each edge carries a relation id.

    expand(src, r) returns the unique tails reachable from any entity in `src`
    along relation r. An optional owner mask restricts which entities may be
    expanded, which is how a client's local view is simulated: a client stores
    exactly the triples whose head it owns (inverse triples included).
    """

    def __init__(self, n_ent: int, heads, rels, tails):
        heads = np.asarray(heads, dtype=np.int64)
        rels = np.asarray(rels, dtype=np.int64)
        tails = np.asarray(tails, dtype=np.int64)
        order = np.lexsort((rels, heads))
        self.n_ent = n_ent
        self.heads = heads[order]
        self.rels = rels[order]
        self.tails = tails[order]
        counts = np.bincount(self.heads, minlength=n_ent)
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    def _edge_idx(self, src: np.ndarray) -> np.ndarray:
        starts = self.indptr[src]
        lens = self.indptr[src + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        offs = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
        return offs + np.arange(total)

    def out_rels(self, src: np.ndarray) -> np.ndarray:
        idx = self._edge_idx(src)
        return np.unique(self.rels[idx]) if idx.size else idx

    def expand(self, src: np.ndarray, r: int) -> np.ndarray:
        idx = self._edge_idx(src)
        if idx.size == 0:
            return idx
        return np.unique(self.tails[idx[self.rels[idx] == r]])

    def expand_scored(self, src: np.ndarray, src_scores: np.ndarray, r: int):
        """Sum-of-scores expansion along r: g(e') = sum over edges f(e). Not clipped."""
        idx = self._edge_idx(src)
        if idx.size == 0:
            return idx, np.zeros(0)
        per_edge_src = np.repeat(np.arange(src.size), self.indptr[src + 1] - self.indptr[src])
        m = self.rels[idx] == r
        t = self.tails[idx[m]]
        s = src_scores[per_edge_src[m]]
        if t.size == 0:
            return t, s
        uniq, inv = np.unique(t, return_inverse=True)
        return uniq, np.bincount(inv, weights=s)


def enumerate_paths(g: RelGraph, e0: int, K: int, owner=None, client=None, max_frontier=200_000,
                    no_return: bool = False):
    """All relation sequences of length K from e0 and their endpoint sets.

    If owner/client are given, only entities owned by `client` are expanded
    (the client's local view); endpoints may be any entity. With no_return, the
    topic entity is removed from every frontier after the first hop (paths may
    not pass back through the topic entity).
    """
    out = {}

    def rec(frontier: np.ndarray, seq: tuple):
        if len(seq) == K:
            out[seq] = frontier
            return
        src = frontier if owner is None else frontier[owner[frontier] == client]
        if src.size == 0:
            return
        for r in g.out_rels(src):
            nxt = g.expand(src, int(r))
            if no_return:
                nxt = nxt[nxt != e0]
            if nxt.size and nxt.size <= max_frontier:
                rec(nxt, seq + (int(r),))

    rec(np.array([e0], dtype=np.int64), ())
    return out


def score_paths(paths: dict, e0: int, answers: np.ndarray, max_keep: int = 3, topic_answer: bool = False):
    """Rank relation sequences by F1 of their endpoints against the answers.

    The topic entity is removed from the endpoints before scoring unless the
    dataset allows it as an answer (PathQuestion). Returns the best F1 and up to
    `max_keep` sequences that reach it, as (seq, P, R).
    """
    ans = set(int(a) for a in answers)
    if not ans:
        return 0.0, []
    scored = []
    for seq, ends in paths.items():
        ends = set(int(x) for x in ends) - (set() if topic_answer else {e0})
        if not ends:
            continue
        hit = len(ends & ans)
        if hit == 0:
            continue
        p, r = hit / len(ends), hit / len(ans)
        scored.append((2 * p * r / (p + r), seq, p, r))
    if not scored:
        return 0.0, []
    best = max(s[0] for s in scored)
    top = sorted((s for s in scored if s[0] >= best - 1e-9), key=lambda s: (-s[3], len(s[1]), s[1]))
    return best, [(list(seq), p, r) for _, seq, p, r in top[:max_keep]]


def follow_path(g: RelGraph, e0: int, seq, owner=None, client=None, no_return: bool = False) -> np.ndarray:
    """Endpoints of one relation sequence, optionally restricted to one client's view."""
    frontier = np.array([e0], dtype=np.int64)
    for r in seq:
        src = frontier if owner is None else frontier[owner[frontier] == client]
        if src.size == 0:
            return np.zeros(0, dtype=np.int64)
        frontier = g.expand(src, int(r))
        if no_return:
            frontier = frontier[frontier != e0]
    return frontier
