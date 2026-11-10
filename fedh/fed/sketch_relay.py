"""Identifier-hiding relay with a trained reasoner (P2).

Per question, with a fresh nonce, the querier first submits the topic entity's token
(score 1), padded to N*L entries. Each relay round t then runs in one of two modes.

push  every client reads its own entities back from the summed sketch (count-min,
      minimum over d rows, clipped to 1, kept if above tau), expands them along
      relations weighted by p_t(r | q), drops partial scores below tau / N, and
      submits the candidates it reached. A candidate reached from several clients
      is submitted several times and summed by the sketch.

pull  every client reads, from the summed sketch, the frontier score of each entity
      that one of its own entities links to (its own and its boundary entities; the
      inverse triples tell it these links), and computes the new score of each of its
      own entities locally: g(x) = min(1, sum_r p_t(r|q) sum_{(x, r^-1, h)} f(h)).
      It submits only its own entities with g >= tau. Every entity is submitted by its
      owner alone, so no token appears twice in a round and sums are exact.

Submissions are capped at L minus the dummy share, padded to exactly L entries
(fixed mass M, or dummies drawn from the real score distribution) and quantised to
1/gamma. Each client keeps a(e) = sum_t p(K = t) f_t(e) for its own entities and
returns exactly k of them to the querier, which ranks the union and drops the topic.
The server only sees the element-wise sum of the d x B sketches.
"""
from __future__ import annotations

import numpy as np

from fedh.fed.execute import FederatedKG
from fedh.relay.sketch import (PadConfig, build_sketch, entity_tokens, pad_entries,
                               randomized_round, read_back)

KEY = 0x2545F4914F6CDD1D


class SketchRelay:
    def __init__(self, kg: FederatedKG, B: int, d: int, pad: PadConfig, gamma: float = 0.0,
                 tau: float = 0.05, k_answers: int = 10, prune: float = 1e-3, mode: str = "push"):
        self.kg, self.B, self.d, self.pad, self.mode = kg, B, d, pad, mode
        self.gamma, self.tau, self.k, self.prune = gamma, tau, k_answers, prune
        self.owned_ids = [np.nonzero(kg.owner == c)[0] for c in range(kg.n_clients)]
        self.owned_tok = [entity_tokens(ids, KEY) for ids in self.owned_ids]
        self.seed_pad = PadConfig(L=pad.L * kg.n_clients, mass_per_slot=pad.mass_per_slot,
                                  min_dummy_frac=pad.min_dummy_frac, dummy_scores=pad.dummy_scores,
                                  beta_a=pad.beta_a, beta_b=pad.beta_b, empirical=pad.empirical,
                                  fixed_mass=pad.fixed_mass)
        if mode == "pull":
            # per client: the entities its stored triples point to (own + boundary), and edge -> tail slot
            self.nbr_ids, self.nbr_tok, self.edge_slot = [], [], []
            for g in kg.graphs:
                u, inv = np.unique(g.tails, return_inverse=True)
                self.nbr_ids.append(u)
                self.nbr_tok.append(entity_tokens(u, KEY))
                self.edge_slot.append(inv)

    # ------------------------------------------------------------------ shared
    def _aggregate(self, entries, cfg, nonce, rng, stats):
        sk = np.zeros((self.d, self.B))
        for ids, sc in entries:
            tok, s, info = pad_entries(entity_tokens(ids, KEY), sc, cfg, rng)
            s = randomized_round(s, self.gamma, rng)
            sk += build_sketch(tok, s, nonce, self.d, self.B)
            if stats is not None:
                stats["n_real"].append(int(ids.size))
                stats["truncated"].append(info.n_truncated)
                stats["scaled"].append(bool(info.scaled))
                if ids.size:
                    stats["real_scores"].append(np.asarray(sc, dtype=np.float32))
        return sk

    # ------------------------------------------------------------------ push
    def _read_own(self, sk, nonce):
        out = []
        for c in range(self.kg.n_clients):
            est = np.minimum(1.0, read_back(sk, self.owned_tok[c], nonce))
            idx = np.nonzero(est > self.tau)[0]
            out.append((self.owned_ids[c][idx], est[idx]))
        return out

    def _push_expand(self, c, ids, sc, rel_probs):
        rels = np.nonzero(rel_probs >= self.prune)[0]
        g = self.kg.graphs[c]
        ti, ts = [], []
        for r in rels:
            t, s = g.expand_scored(ids, sc, int(r))
            if t.size:
                ti.append(t)
                ts.append(s * rel_probs[r])
        if not ti:
            return np.zeros(0, np.int64), np.zeros(0)
        t = np.concatenate(ti)
        uniq, inv = np.unique(t, return_inverse=True)
        sc = np.minimum(1.0, np.bincount(inv, weights=np.concatenate(ts)))
        # a partial score below tau / N cannot lift an entity over tau on its own
        keep = sc >= self.tau / self.kg.n_clients
        return uniq[keep], sc[keep]

    # ------------------------------------------------------------------ pull
    def _pull_scores(self, c, sk, nonce, own_prev, rel_probs):
        """New scores of client c's own entities from the frontier in sketch sk."""
        g = self.kg.graphs[c]
        f = np.minimum(1.0, read_back(sk, self.nbr_tok[c], nonce))
        f[f <= self.tau] = 0.0
        if own_prev is not None:
            # the client knows its own previous scores exactly; only boundary values come from the sketch
            own = self.kg.owner[self.nbr_ids[c]] == c
            f[own] = 0.0
            ids, sc = own_prev
            if ids.size:
                pos = np.searchsorted(self.nbr_ids[c], ids)
                ok = (pos < self.nbr_ids[c].size)
                ok[ok] = self.nbr_ids[c][pos[ok]] == ids[ok]
                f[pos[ok]] = sc[ok]
        fe = f[self.edge_slot[c]]                                  # frontier score at each edge's tail
        w = np.zeros(g.rels.size)
        for r in np.nonzero(rel_probs >= self.prune)[0]:
            m = g.rels == self.kg.inverse_of[r]
            w[m] += rel_probs[r]
        contrib = fe * w
        nz = contrib > 0
        if not nz.any():
            return np.zeros(0, np.int64), np.zeros(0)
        uniq, inv = np.unique(g.heads[nz], return_inverse=True)
        sc = np.minimum(1.0, np.bincount(inv, weights=contrib[nz]))
        keep = sc >= self.tau
        return uniq[keep], sc[keep]

    # ------------------------------------------------------------------ run
    def run(self, e0: int, P: np.ndarray, pK: np.ndarray, rng: np.random.Generator, stats=None,
            keep_sketch=False):
        nonce = int(rng.integers(0, 2**62))
        n = self.kg.n_clients
        sketches = []
        sk = self._aggregate([(np.array([e0], np.int64), np.array([1.0]))], self.seed_pad, nonce, rng, None)
        acc = [dict() for _ in range(n)]
        frontier = self._read_own(sk, nonce) if self.mode == "push" else [None] * n
        for t in range(P.shape[0]):
            if pK[t:].sum() < self.prune:
                break   # the deployed system still sends padded rounds up to K_max; they carry no real mass here
            if self.mode == "push":
                cand = [self._push_expand(c, ids, sc, P[t]) if ids.size else (np.zeros(0, np.int64), np.zeros(0))
                        for c, (ids, sc) in enumerate(frontier)]
            else:
                cand = [self._pull_scores(c, sk, nonce, frontier[c], P[t]) for c in range(n)]
            if self.kg.no_return:
                cand = [(ids[ids != e0], sc[ids != e0]) for ids, sc in cand]
            sk = self._aggregate(cand, self.pad, nonce, rng, stats)
            if keep_sketch:
                sketches.append((sk, sum(int(x[0].size) for x in cand)))
            frontier = self._read_own(sk, nonce) if self.mode == "push" else cand
            if pK[t] >= self.prune:
                for c, (ids, sc) in enumerate(frontier):
                    a = acc[c]
                    for e, s in zip(ids.tolist(), (sc * pK[t]).tolist()):
                        a[e] = a.get(e, 0.0) + s
        ids_out, sc_out = [], []
        for a in acc:
            if not a:
                continue
            e = np.fromiter(a.keys(), np.int64, len(a))
            s = np.fromiter(a.values(), np.float64, len(a))
            top = np.argsort(-s, kind="stable")[: self.k]
            ids_out.append(e[top])
            sc_out.append(s[top])
        if not ids_out:
            return np.zeros(0, np.int64), np.zeros(0), sketches
        e, s = np.concatenate(ids_out), np.concatenate(sc_out)
        if self.kg.topic_answer:
            return e, s, sketches
        m = e != e0
        return e[m], s[m], sketches
