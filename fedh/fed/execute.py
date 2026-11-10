"""Hop execution over a horizontally partitioned KG (exact relay, raw entity IDs).

Each client stores the triples whose head it owns (inverse triples included), so
expanding a frontier entity along any relation needs only its owner. One hop is:

    g(e') = min(1, sum_e sum_r f(e) * p_t(r | q) * A_r(e, e'))

computed by every client on the frontier entities it owns and summed by the relay.
The final answer score mixes hop depths with the hop-count distribution:
a(e) = sum_k p(K = k) * f_k(e). The topic entity is not an answer unless the dataset allows it (stats.json
exec_rules.topic_answer, PathQuestion), and with exec_rules.no_return paths may not
pass back through it.

Modes:
  relay  every client expands what it owns (the federated system, equal to pooled
         execution because each triple lives at the owner of its head)
  local  only the question owner expands; entities it does not own are dead ends
         (the Isolated baseline)
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from fedh.data.graph import RelGraph


class FederatedKG:
    def __init__(self, part_dir: str):
        self.part_dir = part_dir
        with open(os.path.join(part_dir, "entities.tsv"), encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t"))[1:]
        self.owner = np.array([int(r[2]) for r in rows], dtype=np.int64)
        self.n_ent = self.owner.size
        with open(os.path.join(part_dir, "relations.tsv"), encoding="utf-8") as f:
            rels = list(csv.reader(f, delimiter="\t"))[1:]
        self.rel_names = [r[1] for r in rels]
        self.inverse_of = np.array([int(r[2]) for r in rels], dtype=np.int64)
        self.n_rel = len(rels)
        self.n_clients = int(self.owner.max()) + 1
        self.no_return, self.topic_answer = False, False
        sp = os.path.join(part_dir, "stats.json")
        if os.path.exists(sp):
            with open(sp) as f:
                rules = json.load(f).get("exec_rules", {})
            self.no_return = bool(rules.get("no_return", False))
            self.topic_answer = bool(rules.get("topic_answer", False))
        self.graphs = []
        for c in range(self.n_clients):
            npy = os.path.join(part_dir, "clients", f"client_{c}.npy")
            if os.path.exists(npy):
                arr = np.load(npy)
            else:
                arr = np.loadtxt(os.path.join(part_dir, "clients", f"client_{c}.tsv"), dtype=np.int64,
                                 skiprows=1, ndmin=2)
            self.graphs.append(RelGraph(self.n_ent, arr[:, 0], arr[:, 1], arr[:, 2]))

    def out_rels(self, e: int) -> np.ndarray:
        return self.graphs[int(self.owner[e])].out_rels(np.array([e], dtype=np.int64))

    def hop(self, ids, scores, rel_probs, clients, prune=1e-3, cap=0):
        """One hop of sum-and-clip scoring. Returns (ids, scores) of the next frontier."""
        rels = np.nonzero(rel_probs >= prune)[0]
        out_i, out_s = [], []
        for c in clients:
            m = self.owner[ids] == c
            if not m.any():
                continue
            src, ssc = ids[m], scores[m]
            g = self.graphs[c]
            for r in rels:
                t, s = g.expand_scored(src, ssc, int(r))
                if t.size:
                    out_i.append(t)
                    out_s.append(s * rel_probs[r])
        if not out_i:
            return np.zeros(0, np.int64), np.zeros(0)
        t = np.concatenate(out_i)
        s = np.concatenate(out_s)
        uniq, inv = np.unique(t, return_inverse=True)
        agg = np.minimum(1.0, np.bincount(inv, weights=s))
        if cap:
            keep = []
            own = self.owner[uniq]
            for c in np.unique(own):
                idx = np.nonzero(own == c)[0]
                if idx.size > cap:
                    idx = idx[np.argsort(-agg[idx], kind="stable")[:cap]]
                keep.append(idx)
            keep = np.sort(np.concatenate(keep))
            uniq, agg = uniq[keep], agg[keep]
        return uniq, agg

    def answer_scores(self, e0: int, P: np.ndarray, pK: np.ndarray, mode: str = "relay",
                      prune: float = 1e-3, cap: int = 0):
        """P: [Kmax, n_rel] per-hop relation probabilities; pK: [Kmax] hop-count probabilities."""
        clients = range(self.n_clients) if mode == "relay" else [int(self.owner[e0])]
        ids = np.array([e0], dtype=np.int64)
        sc = np.ones(1)
        total = {}
        for t in range(P.shape[0]):
            if pK[t:].sum() < prune or ids.size == 0:
                break
            ids, sc = self.hop(ids, sc, P[t], clients, prune, cap)
            if self.no_return and ids.size:
                keep = ids != e0
                ids, sc = ids[keep], sc[keep]
            if pK[t] >= prune:
                for e, s in zip(ids.tolist(), (sc * pK[t]).tolist()):
                    total[e] = total.get(e, 0.0) + s
        if not self.topic_answer:
            total.pop(int(e0), None)
        if not total:
            return np.zeros(0, np.int64), np.zeros(0)
        e = np.fromiter(total.keys(), dtype=np.int64, count=len(total))
        s = np.fromiter(total.values(), dtype=np.float64, count=len(total))
        return e, s


def sparsify(P: np.ndarray, pK: np.ndarray, rel: float = 0.0, top: int = 0):
    """Keep, per hop, relations with p >= rel * max_r p (at most `top` of them) and renormalise;
    same for the hop count. rel = 0 and top = 0 leave the distributions unchanged.
    Every client computes this from the shared model output, so it reveals nothing new."""
    if rel <= 0 and top <= 0:
        return P, pK
    P = P.copy()
    for t in range(P.shape[0]):
        row = P[t]
        keep = row >= rel * row.max()
        if top > 0 and keep.sum() > top:
            keep[:] = False
            keep[np.argsort(-row)[:top]] = True
        row[~keep] = 0.0
        P[t] = row / max(row.sum(), 1e-12)
    pK = pK.copy()
    keep = pK >= rel * pK.max()
    pK[~keep] = 0.0
    return P, pK / max(pK.sum(), 1e-12)


def hits_at_1(ids: np.ndarray, scores: np.ndarray, answers) -> float:
    """Expected Hits@1 under random tie-breaking among the top-scored entities."""
    if ids.size == 0:
        return 0.0
    top = scores.max()
    tied = ids[scores >= top - 1e-9]
    ans = set(int(a) for a in answers)
    return sum(1 for x in tied.tolist() if x in ans) / tied.size


def follow_path_fed(kg: FederatedKG, e0: int, seq, mode: str = "relay") -> np.ndarray:
    """Endpoints of one relation sequence through the exact relay (or the owner only)."""
    P = np.zeros((len(seq), kg.n_rel))
    for t, r in enumerate(seq):
        P[t, r] = 1.0
    pK = np.zeros(len(seq))
    pK[-1] = 1.0
    ids, _ = kg.answer_scores(e0, P, pK, mode=mode)
    return ids
