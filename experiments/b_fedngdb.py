"""FedNGDB (Hu et al., 2024) adapted to multi-hop KGQA on the FedH-KGQA partitions.

FedNGDB answers complex logical queries over distributed KGs with a federated neural graph database:
entity embeddings are combined across clients by secret aggregation (the server sees their average
over the clients that hold each entity; it needs the shared entity index to do so), the operator
networks are averaged with FedAvg, and each client scores its own entities against the query
encoding; the server merges the score tables.

Adaptation. FedNGDB takes structured queries, not questions, so the relation path of each test
question comes from FedH-KGQA's own reasoner (top hop count, top relation per hop, from its saved
test_probs.npz). This isolates the execution step: embedding-based query answering (FedNGDB) against
FedH-KGQA's symbolic relay on the same predicted path. The query encoder is the GQE-style point
encoder with translation projections, q = e_topic + sum_t r_t, trained on path queries of 1 to K_max
hops sampled inside each client's own graph (in-graph queries, as in FedNGDB training), scored by
-||q - e||. Entity embeddings are averaged over their holders and relation (operator) embeddings
with FedAvg after every round.

  python experiments/b_fedngdb.py --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \\
      --probs results/p3v2/metaqa2/<run>/test_probs.npz
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_kgqa import Partition, fmt, read_jsonl, score  # noqa: E402


class Client:
    def __init__(self, part, c, d, k_max, full, lr, device, rng):
        arr, g2l = part.triples[c], part.g2l[c]
        self.n = part.view[c].size
        self.H = g2l[arr[:, 0]]
        self.R = arr[:, 1].astype(np.int64)
        self.T = g2l[arr[:, 2]]
        own = np.zeros(self.n, dtype=bool)
        own[g2l[part.owned_g[c]]] = True
        self.own = own
        order = np.argsort(self.H, kind="stable")
        self.H, self.R, self.T = self.H[order], self.R[order], self.T[order]
        self.ptr = np.searchsorted(self.H, np.arange(self.n + 1))       # out-edges of local entity i
        self.k_max, self.full, self.device, self.rng = k_max, full, device, rng
        self.ent = torch.nn.Parameter(torch.randn(self.n, d, device=device) * (1e-3 if full else 0.1 / d ** 0.5))
        self.lr = lr
        self.relcount = np.bincount(self.R, minlength=part.n_rel)

    def sample(self, m):
        """m path queries of 1..k_max hops inside this client's graph: (start, relation path, answer)."""
        e = self.rng.integers(0, self.H.size, m)
        start, cur = self.H[e], self.T[e]
        rels = np.full((m, self.k_max), -1, dtype=np.int64)
        rels[:, 0] = self.R[e]
        want = self.rng.integers(1, self.k_max + 1, m)
        for t in range(1, self.k_max):
            deg = self.ptr[cur + 1] - self.ptr[cur]
            go = (want > t) & self.own[cur] & (deg > 0)
            idx = np.nonzero(go)[0]
            if idx.size == 0:
                break
            pick = self.ptr[cur[idx]] + (self.rng.random(idx.size) * deg[idx]).astype(np.int64)
            rels[idx, t] = self.R[pick]
            cur[idx] = self.T[pick]
        return start, rels, cur

    def train(self, rel, epochs, batch, n_neg):
        opt = torch.optim.Adagrad([self.ent, rel], lr=self.lr) if self.full else torch.optim.Adam([self.ent, rel], lr=self.lr)
        tot, cnt = 0.0, 0
        for _ in range(epochs):
            for _ in range(max(1, self.H.size // batch)):
                s, rp, a = self.sample(batch)
                s, a = torch.as_tensor(s, device=self.device), torch.as_tensor(a, device=self.device)
                rp = torch.as_tensor(rp, device=self.device)
                q = self.ent[s] + (rel[rp.clamp(min=0)] * (rp >= 0).unsqueeze(-1)).sum(1)
                if self.full:
                    logits = -torch.cdist(q, self.ent)
                    loss = F.cross_entropy(logits, a)
                else:
                    neg = self.ent[torch.randint(0, self.n, (n_neg,), device=self.device)]
                    pos = -(q - self.ent[a]).norm(dim=-1, keepdim=True)
                    loss = F.cross_entropy(torch.cat([pos, -torch.cdist(q, neg)], 1),
                                           torch.zeros(len(a), dtype=torch.long, device=self.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * batch
                cnt += batch
        return tot / max(1, cnt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--probs", required=True, help="FedH-KGQA test_probs.npz (relation paths of the test questions)")
    ap.add_argument("--dim", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--neg", type=int, default=4096)
    ap.add_argument("--full-max", type=int, default=300_000)
    ap.add_argument("--lr-full", type=float, default=0.1, help="Adagrad (full softmax)")
    ap.add_argument("--lr-sampled", type=float, default=1e-3, help="Adam (sampled softmax)")
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    part_name = os.path.basename(os.path.normpath(args.data_dir))
    hops = os.path.basename(os.path.dirname(os.path.normpath(args.data_dir)))
    tag = args.tag or f"{hops}_{part_name}_fedngdb_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    part = Partition(args.data_dir)
    z = np.load(args.probs)
    pos = {q: i for i, q in enumerate(z["qid"].tolist())}
    test = [q for q in read_jsonl(os.path.join(args.data_dir, "test.jsonl")) if q["qid"] in pos]
    k_max = z["P"].shape[1]
    paths = []
    for q in test:
        P, pK = z["P"][pos[q["qid"]]].astype(np.float32), z["pK"][pos[q["qid"]]]
        k = int(pK.argmax()) + 1
        paths.append([int(P[t].argmax()) for t in range(k)])
    print(f"[data] {part_name} ({hops}): {len(test)} test questions with a predicted path; {part.n_clients} clients, "
          f"{part.n_ent} entities, {part.n_rel} relations; K_max {k_max}; device {device}")

    clients = []
    for c in range(part.n_clients):
        full = part.view[c].size <= args.full_max
        clients.append(Client(part, c, args.dim, k_max, full, args.lr_full if full else args.lr_sampled, device, rng))
    print(f"[fedngdb] {'full' if clients[0].full else 'sampled'} softmax; entities per client "
          f"{[cl.n for cl in clients]}")
    views = [torch.as_tensor(v, device=device) for v in part.view]
    rels = [torch.nn.Parameter(torch.randn(part.n_rel, args.dim, device=device) * (1e-3 if cl.full else 0.1 / args.dim ** 0.5))
            for cl in clients]
    rc = torch.as_tensor(np.stack([cl.relcount for cl in clients]), dtype=torch.float32, device=device)
    rw = rc / rc.sum(0, keepdim=True).clamp(min=1)

    def aggregate():
        with torch.no_grad():
            S = torch.zeros((part.n_ent, args.dim), device=device)
            N = torch.zeros(part.n_ent, device=device)
            for c, cl in enumerate(clients):
                S.index_add_(0, views[c], cl.ent)
                N.index_add_(0, views[c], torch.ones(views[c].numel(), device=device))
            S /= N.clamp(min=1).unsqueeze(1)
            for c, cl in enumerate(clients):
                cl.ent.copy_(S[views[c]])
            R = sum(rw[c].unsqueeze(1) * rels[c] for c in range(len(clients)))
            for r in rels:
                r.copy_(R)

    aggregate()
    log = []
    for rd in range(args.rounds):
        ts = time.time()
        losses = [cl.train(rels[c], args.local_epochs, args.batch_size, args.neg) for c, cl in enumerate(clients)]
        aggregate()
        log.append({"round": rd + 1, "loss": losses})
        print(f"[round {rd + 1}/{args.rounds}] loss {np.round(losses, 4).tolist()} ({time.time() - ts:.0f}s)")

    # answer: the topic owner sends e_topic; every client scores the entities it owns; best overall
    with torch.no_grad():
        R = rels[0]
        topics = np.array([q["topic_id"] for q in test])
        owner = part.owner[topics]
        Q = torch.zeros((len(test), args.dim), device=device)
        for o, cl in enumerate(clients):
            m = np.nonzero(owner == o)[0]
            if m.size:
                Q[torch.as_tensor(m, device=device)] = cl.ent[torch.as_tensor(part.g2l[o][topics[m]], device=device)]
        for i, p in enumerate(paths):
            Q[i] += R[torch.as_tensor(p, device=device)].sum(0)
        best = torch.full((len(test),), float("-inf"), device=device)
        best_id = torch.full((len(test),), -1, dtype=torch.long, device=device)
        tq = torch.as_tensor(topics, device=device)
        for c, cl in enumerate(clients):
            loc = torch.as_tensor(part.owned_l[c], device=device)
            gid = torch.as_tensor(part.owned_g[c], device=device)
            Ec = cl.ent[loc]
            for i in range(0, len(test), args.eval_batch_size):
                s = -torch.cdist(Q[i: i + args.eval_batch_size], Ec)
                if not part.topic_answer:
                    s = s.masked_fill(gid.unsqueeze(0) == tq[i: i + args.eval_batch_size].unsqueeze(1), float("-inf"))
                v, a = s.max(1)
                upd = v > best[i: i + args.eval_batch_size]
                best[i: i + args.eval_batch_size] = torch.where(upd, v, best[i: i + args.eval_batch_size])
                best_id[i: i + args.eval_batch_size] = torch.where(upd, gid[a], best_id[i: i + args.eval_batch_size])
    top = best_id.cpu().numpy()
    tm, hits = score(test, top)
    print(f"[test] {fmt(tm)} (n={tm['n']}, cross {tm['n_cross']})")
    res = {"tag": tag, "method": "fedngdb", "args": vars(args), "data": part_name, "hops": hops, "history": log,
           "test": tm, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "test_preds.jsonl"), "w") as f:
        for q, t, h in zip(test, top, hits):
            f.write(json.dumps({"qid": q["qid"], "top": int(t), "hit": float(h), "cross": bool(q.get("cross_client"))}) + "\n")
    print(f"[done] {out_dir}/metrics.json in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
