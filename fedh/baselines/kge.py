"""ComplEx and TransE knowledge-graph embeddings, trained centrally or per client (FedE / FedR).

ComplEx: score(h, r, t) = Re(<h, r, conj(t)>).   TransE: score(h, r, t) = -||h + r - t||.

Every client stores the triples whose head it owns, inverse triples included, so tail
prediction over its own triples covers both directions (reciprocal training). Graphs up to a
few hundred thousand entities train with a full softmax over all entities (1-vs-all, N3
regularisation, Adagrad), the standard strong recipe for ComplEx. Larger graphs (WebQSP,
1.3M entities) use sampled softmax with sparse embeddings and SparseAdam.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def complex_hr(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """(h * r) in ComplEx; score(h, r, t) = <complex_hr(h, r), t> = Re(<h, r, conj(t)>)."""
    k = h.shape[-1] // 2
    a, b = h[..., :k], h[..., k:]
    c, d = r[..., :k], r[..., k:]
    return torch.cat([a * c - b * d, a * d + b * c], -1)


def compose(kind: str, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """The query vector q(h, r); score(h, r, t) = match(kind, q, t)."""
    return complex_hr(h, r) if kind == "complex" else h + r


def match(kind: str, q: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Scores of query vectors q [B, d] against entity embeddings E [N, d] -> [B, N]."""
    return q @ E.t() if kind == "complex" else -torch.cdist(q, E)


class ComplEx(nn.Module):
    """KGE model; kind = complex (default) or transe (same embedding size, 2 x rank)."""

    def __init__(self, n_ent: int, n_rel: int, rank: int = 200, init: float = 1.0, sparse: bool = True,
                 kind: str = "complex"):
        super().__init__()
        self.rank, self.kind = rank, kind
        self.ent = nn.Embedding(n_ent, 2 * rank, sparse=sparse)
        self.rel = nn.Embedding(n_rel, 2 * rank, sparse=sparse)
        nn.init.normal_(self.ent.weight, std=init / rank ** 0.5)
        nn.init.normal_(self.rel.weight, std=init / rank ** 0.5)

    def _n3(self, *xs):
        if self.kind != "complex":
            return sum((x ** 2).sum(-1).mean() for x in xs)      # plain L2 for TransE
        k = self.rank
        return sum((x[..., :k] ** 2 + x[..., k:] ** 2).sqrt().pow(3).sum(-1).mean() for x in xs)

    def loss(self, h, r, t, n_ent_sample: int, n_neg: int, reg: float):
        """Sampled softmax: the true tail against n_neg uniformly drawn tails (large graphs)."""
        he, re, te = self.ent(h), self.rel(r), self.ent(t)
        q = compose(self.kind, he, re)
        pos = (q * te).sum(-1, keepdim=True) if self.kind == "complex" else -(q - te).norm(dim=-1, keepdim=True)
        neg_ids = torch.randint(0, n_ent_sample, (n_neg,), device=h.device)
        neg = match(self.kind, q, self.ent(neg_ids))
        logits = torch.cat([pos, neg], 1)
        ce = F.cross_entropy(logits, torch.zeros(len(h), dtype=torch.long, device=h.device))
        return ce + reg * self._n3(he, re, te), ce.detach()

    def loss_full(self, h, r, t, reg: float):
        """Full softmax over every entity (1-vs-all, as in KBC with N3; small and medium graphs)."""
        he, re, te = self.ent(h), self.rel(r), self.ent(t)
        logits = match(self.kind, compose(self.kind, he, re), self.ent.weight)
        ce = F.cross_entropy(logits, t)
        return ce + reg * self._n3(he, re, te), ce.detach()


class KGETrainer:
    """Holds one ComplEx model and its triples (local indices) and runs local epochs.

    full=True: full softmax, dense embeddings and Adagrad (Lacroix et al., 2018), init 1e-3.
    full=False: sampled softmax, sparse embeddings and SparseAdam (graphs too large for 1-vs-all)."""

    def __init__(self, n_ent, n_rel, heads, rels, tails, rank, lr, device, seed=0, init=1.0, full=False,
                 kind="complex"):
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
        self.full = full
        if full:
            init = 1e-3 * rank ** 0.5     # absolute std 1e-3
        self.kind = kind
        self.model = ComplEx(n_ent, n_rel, rank, init, sparse=not full, kind=kind).to(device)
        self.H = torch.as_tensor(np.asarray(heads), dtype=torch.long, device=device)
        self.R = torch.as_tensor(np.asarray(rels), dtype=torch.long, device=device)
        self.T = torch.as_tensor(np.asarray(tails), dtype=torch.long, device=device)
        self.n_ent, self.lr, self.device = n_ent, lr, device
        self.prox, self.beta = None, 0.0     # PFedEG: pull entity embeddings toward personalized ones
        self.reset_optimizer()

    def reset_optimizer(self):
        params = list(self.model.parameters())
        self.opt = torch.optim.Adagrad(params, lr=self.lr) if self.full else torch.optim.SparseAdam(params, lr=self.lr)

    def train_epochs(self, epochs, batch=4096, n_neg=1024, reg=1e-3):
        n = self.H.numel()
        tot, cnt = 0.0, 0
        if n == 0:
            return float("nan")
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch):
                idx = perm[i: i + batch]
                if self.full:
                    loss, ce = self.model.loss_full(self.H[idx], self.R[idx], self.T[idx], reg)
                else:
                    loss, ce = self.model.loss(self.H[idx], self.R[idx], self.T[idx], self.n_ent, n_neg, reg)
                if self.prox is not None and self.beta > 0:
                    rows = torch.cat([self.H[idx], self.T[idx]])
                    loss = loss + self.beta * (self.model.ent(rows) - self.prox[rows]).norm()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                tot += float(ce) * idx.numel()
                cnt += idx.numel()
        return tot / max(1, cnt)

    @torch.no_grad()
    def tail_mrr(self, n: int = 2000, seed: int = 0):
        """Raw tail-prediction MRR and Hits@10 on a sample of the training triples (a fit check)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = self.H.numel()
        if m == 0:
            return 0.0, 0.0
        idx = torch.randint(0, m, (min(n, m),), generator=g).to(self.device)
        E = self.model.ent.weight
        q = compose(self.kind, E[self.H[idx]], self.model.rel.weight[self.R[idx]])
        ranks = []
        for i in range(0, idx.numel(), 256):
            s = match(self.kind, q[i: i + 256], E)
            tgt = s.gather(1, self.T[idx[i: i + 256]].unsqueeze(1))
            ranks.append((s >= tgt).sum(1))  # ties count against the target
        r = torch.cat(ranks).float()
        return float((1.0 / r).mean()), float((r <= 10).float().mean())


# ----------------------------------------------------------------------------- PFLU (anchor-based)

def select_anchors(n_ent, heads, tails, rels, ratio=0.1, k=20, m=15, n_rel=0, candidates=None):
    """PFLU-style anchors for one client (local ids): the top `ratio` of its entities by degree, taken
    from `candidates` (entities the client shares with others, PFLU's overlap anchors) when given.
    Each entity is encoded from up to k anchors among itself and its neighbours (highest degree
    first) and up to m relations on its edges. Returns anchors, entity2anchor [n, k] (pad = n_anchor),
    entity2rel [n, m] (pad = n_rel)."""
    heads, tails, rels = (np.asarray(x, dtype=np.int64) for x in (heads, tails, rels))
    deg = np.bincount(np.concatenate([heads, tails]), minlength=n_ent)
    n_anchor = max(1, int(np.ceil(ratio * n_ent)))
    pool = np.arange(n_ent) if candidates is None else np.nonzero(candidates)[0]
    anchors = pool[np.argsort(-deg[pool], kind="stable")[:n_anchor]]
    n_anchor = anchors.size
    a_idx = np.full(n_ent, -1, dtype=np.int64)
    a_idx[anchors] = np.arange(n_anchor)
    ent = np.concatenate([heads, tails, anchors])
    nb = np.concatenate([tails, heads, anchors])          # neighbours both ways, plus each anchor itself
    keep = a_idx[nb] >= 0
    ent, nb = ent[keep], nb[keep]
    pair = np.unique(np.stack([ent, nb], 1), axis=0)
    ent, nb = pair[:, 0], pair[:, 1]
    order = np.lexsort((-deg[nb], ent))
    ent, nb = ent[order], nb[order]
    start = np.searchsorted(ent, ent, side="left")
    rank = np.arange(ent.size) - start
    sel = rank < k
    e2a = np.full((n_ent, k), n_anchor, dtype=np.int64)
    e2a[ent[sel], rank[sel]] = a_idx[nb[sel]]
    er = np.unique(np.stack([np.concatenate([heads, tails]), np.concatenate([rels, rels])], 1), axis=0)
    start = np.searchsorted(er[:, 0], er[:, 0], side="left")
    rank = np.arange(er.shape[0]) - start
    sel = rank < m
    e2r = np.full((n_ent, m), n_rel, dtype=np.int64)
    e2r[er[sel, 0], rank[sel]] = er[sel, 1]
    return anchors, e2a, e2r


class AnchorKGE(nn.Module):
    """Entity embedding = MLP([mean of its anchors' embeddings ; mean of its relations' embeddings]);
    an anchor entity uses its anchor embedding directly. Only anchor embeddings are federated."""

    def __init__(self, n_ent, n_rel, rank, anchors, e2a, e2r):
        super().__init__()
        d = 2 * rank
        self.rank, self.kind = rank, "complex"
        self.n_anchor = anchors.size
        self.anchor = nn.Parameter(torch.randn(self.n_anchor + 1, d) * 0.1 / rank ** 0.5)
        self.rel = nn.Embedding(n_rel + 1, d)
        nn.init.normal_(self.rel.weight, std=0.1 / rank ** 0.5)
        self.mlp = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.register_buffer("e2a", torch.as_tensor(e2a), persistent=False)
        self.register_buffer("e2r", torch.as_tensor(e2r), persistent=False)
        a_of = torch.full((n_ent,), -1, dtype=torch.long)
        a_of[torch.as_tensor(anchors)] = torch.arange(self.n_anchor)
        self.register_buffer("a_of", a_of, persistent=False)

    def entities(self, ids):
        A = torch.cat([self.anchor[:-1], torch.zeros_like(self.anchor[-1:])])   # padding row is zero
        ia = self.e2a[ids]
        ma = (ia < self.n_anchor).float().unsqueeze(-1)
        ea = (A[ia] * ma).sum(1) / ma.sum(1).clamp(min=1)
        ir = self.e2r[ids]
        mr = (ir < self.rel.num_embeddings - 1).float().unsqueeze(-1)
        er = (self.rel(ir) * mr).sum(1) / mr.sum(1).clamp(min=1)
        e = self.mlp(torch.cat([ea, er], -1))
        ao = self.a_of[ids]
        return torch.where((ao >= 0).unsqueeze(-1), A[ao.clamp(min=0)], e)


class AnchorTrainer:
    """PFLU-style client: trains anchor embeddings, relation embeddings and the encoder locally."""

    def __init__(self, n_ent, n_rel, heads, rels, tails, rank, lr, device, seed=0, full=False,
                 ratio=0.1, k=20, m=15, candidates=None):
        torch.manual_seed(seed)
        self.full, self.kind, self.n_ent, self.device = full, "complex", n_ent, device
        self.anchors, e2a, e2r = select_anchors(n_ent, heads, tails, rels, ratio, k, m, n_rel, candidates)
        self.model = AnchorKGE(n_ent, n_rel, rank, self.anchors, e2a, e2r).to(device)
        self.H = torch.as_tensor(np.asarray(heads), dtype=torch.long, device=device)
        self.R = torch.as_tensor(np.asarray(rels), dtype=torch.long, device=device)
        self.T = torch.as_tensor(np.asarray(tails), dtype=torch.long, device=device)
        self.lr = lr
        self.reset_optimizer()

    def reset_optimizer(self):
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epochs(self, epochs, batch=4096, n_neg=1024, reg=0.0):
        n = self.H.numel()
        tot, cnt = 0.0, 0
        all_ids = torch.arange(self.n_ent, device=self.device)
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, batch):
                idx = perm[i: i + batch]
                h = self.model.entities(self.H[idx])
                q = compose("complex", h, self.model.rel(self.R[idx]))
                if self.full:
                    logits = match("complex", q, self.model.entities(all_ids))
                    ce = F.cross_entropy(logits, self.T[idx])
                else:
                    t = self.model.entities(self.T[idx])
                    neg = self.model.entities(torch.randint(0, self.n_ent, (n_neg,), device=self.device))
                    logits = torch.cat([(q * t).sum(-1, keepdim=True), match("complex", q, neg)], 1)
                    ce = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=self.device))
                self.opt.zero_grad(set_to_none=True)
                ce.backward()
                self.opt.step()
                tot += ce.item() * idx.numel()
                cnt += idx.numel()
        return tot / max(1, cnt)

    @torch.no_grad()
    def table(self, chunk=65536):
        return torch.cat([self.model.entities(torch.arange(i, min(i + chunk, self.n_ent), device=self.device))
                          for i in range(0, self.n_ent, chunk)])

    @torch.no_grad()
    def tail_mrr(self, n: int = 2000, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        m = self.H.numel()
        idx = torch.randint(0, m, (min(n, m),), generator=g).to(self.device)
        E = self.table()
        q = compose("complex", E[self.H[idx]], self.model.rel(self.R[idx]))
        ranks = []
        for i in range(0, idx.numel(), 256):
            s = match("complex", q[i: i + 256], E)
            ranks.append((s >= s.gather(1, self.T[idx[i: i + 256]].unsqueeze(1))).sum(1))
        r = torch.cat(ranks).float()
        return float((1.0 / r).mean()), float((r <= 10).float().mean())
