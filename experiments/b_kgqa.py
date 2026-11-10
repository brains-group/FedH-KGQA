"""KGE-based baselines on the FedH-KGQA partitions: EmbedKGQA, FedE + EmbedKGQA head, FedR + EmbedKGQA head.

All three answer a question by scoring entities with ComplEx: score(a) = Re(<e_topic, q, conj(e_a)>),
where q is a relation-like vector produced from the masked question by the same encoder FedH-KGQA
uses (roberta-base with LoRA). They differ in how the embeddings are trained and where scoring runs.

  embedkgqa  Centralized (Saxena et al., 2020): one ComplEx model on the pooled KG, one QA model on
             all training questions, every entity scored. An external centralized reference.
  fede       FedE (Chen et al., 2021): each client trains ComplEx on its own triples; every round the
             server averages the embeddings of entities held by several clients (it needs the shared
             entity IDs to do so) and the relation embeddings.
  fedr       FedR (Zhang et al., 2022): as FedE, but only relation embeddings are aggregated; entity
             embeddings never leave the client, so a boundary entity has a different embedding at
             each client that holds it.

  flkgqa     FL-KG-QA (Gunti et al., JISEM 2025): FedR-style federated TransE (only relation
             embeddings are aggregated, under secure aggregation) and KEQA-style QA run by each client
             on its own: a client-local predicate-learning model regresses the question onto the
             relation embedding of its label (for k hops, the sum of the path's relation embeddings,
             TransE's composition), and the answer is the entity nearest to e_topic + r_q among the
             entities the topic owner stores. QA models are not shared and answering never leaves the
             owner, as in the paper. The topic entity is given, so KEQA's head-detection step is skipped.
             Labels: the owner's own full relation paths (answer F1 >= --local-full-f1).

  pfedeg     PFedEG (Zhang et al., 2024): clients send entity embeddings; the server builds a client-wise
             relation graph (PFedEG*: W_ij = shared-entity Jaccard, W_ii = min_j W_ij) and returns each
             client personalized embeddings K_c = p * weighted mean + (1 - p) * own; clients start from K_c
             and train with a proximal term beta * ||E_c - K_c||. Relation embeddings stay local.
  pflu       PFLU (ISWC 2025, anchor-based parameter-efficient federated KGE): each client encodes an entity
             as MLP([mean of its anchors ; mean of its relations]); anchors are a global set of entities stored
             by two or more clients (PFLU's overlap anchors): the most connected 10% (PFLU uses greedy
             coverage), each linked to entities within one hop; only anchor
             embeddings are averaged across clients. Relations and the encoder stay local.

Federated QA (fede, fedr, pfedeg, pflu): the QA model (LoRA + projection) is trained with FedAvg; each client
trains on the questions it owns and scores only the entities it stores (own + boundary), with its
own embeddings. At test time the topic owner sends the topic embedding to every client, each
client scores the entities it owns, and the best-scoring candidate over all clients is the answer.

  python experiments/b_kgqa.py --method fede --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.baselines.kge import AnchorTrainer, KGETrainer, compose, match  # noqa: E402
from fedh.model.reasoner import BowEncoder, BowTokenizer, HFEncoder  # noqa: E402


# ----------------------------------------------------------------------------- data

def read_jsonl(path, limit=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


class Partition:
    def __init__(self, d):
        with open(os.path.join(d, "entities.tsv"), encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter="\t"))[1:]
        self.owner = np.array([int(r[2]) for r in rows], dtype=np.int64)
        self.n_ent = self.owner.size
        with open(os.path.join(d, "relations.tsv"), encoding="utf-8") as f:
            self.n_rel = len(list(csv.reader(f, delimiter="\t"))) - 1
        self.n_clients = int(self.owner.max()) + 1
        rules = {}
        if os.path.exists(os.path.join(d, "stats.json")):
            with open(os.path.join(d, "stats.json")) as f:
                rules = json.load(f).get("exec_rules", {})
        self.topic_answer = bool(rules.get("topic_answer", False))
        self.triples, self.view, self.g2l, self.owned_l, self.owned_g = [], [], [], [], []
        for c in range(self.n_clients):
            npy = os.path.join(d, "clients", f"client_{c}.npy")
            arr = np.load(npy) if os.path.exists(npy) else np.loadtxt(
                os.path.join(d, "clients", f"client_{c}.tsv"), dtype=np.int64, skiprows=1, ndmin=2)
            owned = np.nonzero(self.owner == c)[0]
            view = np.unique(np.concatenate([owned, arr[:, 0], arr[:, 2]]))
            g2l = np.full(self.n_ent, -1, dtype=np.int64)
            g2l[view] = np.arange(view.size)
            self.triples.append(arr)
            self.view.append(view)
            self.g2l.append(g2l)
            self.owned_g.append(owned)
            self.owned_l.append(g2l[owned])


# ----------------------------------------------------------------------------- QA model

class QAModel(nn.Module):
    def __init__(self, encoder, rank, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * rank))

    def forward(self, ids, att):
        return self.proj(self.encoder(ids, att).float())

    def trainable_state(self):
        return {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, st):
        params = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in st.items():
                params[n].copy_(v)


def fedavg(states, weights):
    tot = float(sum(weights))
    return {k: sum(s[k].float() * (w / tot) for s, w in zip(states, weights)).to(states[0][k].dtype) for k in states[0]}


class Tok:
    def __init__(self, args, texts):
        self.kind, self.max_len = args.encoder, args.max_len
        if self.kind == "bow":
            self.tok = BowTokenizer.build([t.replace("[ENT]", "[ent]") for t in texts])
            self.pad_id = 0
        else:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(args.model_name)
            self.pad_id = self.tok.pad_token_id

    def encode(self, qs):
        if self.kind == "bow":
            return [self.tok.encode(q["question_masked"].replace("[ENT]", "[ent]"), self.max_len) for q in qs]
        texts = [q["question_masked"].replace("[ENT]", self.tok.mask_token) for q in qs]
        return self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]


def pad_batch(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids.to(device), att.to(device)


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def make_examples(qs, ids, g2l, topic_answer):
    """Questions whose topic and at least one answer the trainer knows; targets in its index space."""
    ex = []
    for q, x in zip(qs, ids):
        t = int(g2l[q["topic_id"]])
        ans = [int(g2l[a]) for a in q["answers"] if g2l[a] >= 0 and (topic_answer or a != q["topic_id"])]
        if t >= 0 and ans:
            ex.append({"ids": x, "topic": t, "ans": sorted(set(ans))})
    return ex


def train_qa(model, ex, E, epochs, args, device, pad_id, rng, topic_answer, kind="complex"):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": args.lr, "weight_decay": 0.01}]
    if E.requires_grad:      # EmbedKGQA fine-tunes entity embeddings on MetaQA (--freeze 0)
        groups.append({"params": [E], "lr": args.ent_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    drop_e, drop_q = nn.Dropout(args.ent_drop), nn.Dropout(args.rel_drop)
    steps = max(1, math.ceil(len(ex) / args.batch_size) * epochs)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
    tot, n = 0.0, 0
    for _ in range(epochs):
        order = list(range(len(ex)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            b = [ex[j] for j in order[i: i + args.batch_size]]
            ids, att = pad_batch([e["ids"] for e in b], pad_id, device)
            topic = torch.tensor([e["topic"] for e in b], device=device)
            with autocast(device):
                qv = model(ids, att)
            s = match(kind, compose(kind, drop_e(E[topic]), drop_q(qv.float())), E)
            if not topic_answer:
                s = s.index_put((torch.arange(len(b), device=device), topic), torch.tensor(float("-inf"), device=device))
            tgt = torch.zeros_like(s)
            for k, e in enumerate(b):
                tgt[k, e["ans"]] = 1.0 / len(e["ans"])
            loss = -(tgt * torch.log_softmax(s, -1).clamp(min=-1e4)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            n += len(b)
    return tot / max(1, n)


@torch.no_grad()
def encode_questions(model, ids, args, device, pad_id):
    model.eval()
    out = []
    for i in range(0, len(ids), args.eval_batch_size):
        x, att = pad_batch(ids[i: i + args.eval_batch_size], pad_id, device)
        with autocast(device):
            out.append(model(x, att).float())
    return torch.cat(out) if out else torch.zeros(0)


@torch.no_grad()
def answer(model, qs, ids, tables, part, central, args, device, pad_id, kind="complex"):
    """Top-1 entity per question (global id). central: one table over all entities."""
    qv = encode_questions(model, ids, args, device, pad_id)
    topics = np.array([q["topic_id"] for q in qs], dtype=np.int64)
    top = np.full(len(qs), -1, dtype=np.int64)
    bs = args.eval_batch_size
    if central:
        E = tables[0]
        for i in range(0, len(qs), bs):
            t = torch.as_tensor(topics[i: i + bs], device=device)
            s = match(kind, compose(kind, E[t], qv[i: i + bs]), E)
            if not part.topic_answer:
                s[torch.arange(len(t), device=device), t] = float("-inf")
            top[i: i + bs] = s.argmax(1).cpu().numpy()
        return top
    # topic embedding from its owner's table
    owner = part.owner[topics]
    H = torch.zeros((len(qs), qv.shape[1]), device=device)
    for o in range(part.n_clients):
        m = np.nonzero(owner == o)[0]
        if m.size:
            loc = torch.as_tensor(part.g2l[o][topics[m]], device=device)
            H[torch.as_tensor(m, device=device)] = tables[o][loc]
    best = torch.full((len(qs),), float("-inf"), device=device)
    best_id = torch.full((len(qs),), -1, dtype=torch.long, device=device)
    for c in range(part.n_clients):
        if part.owned_l[c].size == 0:
            continue
        Ec = tables[c][torch.as_tensor(part.owned_l[c], device=device)]
        gid = torch.as_tensor(part.owned_g[c], device=device)
        for i in range(0, len(qs), bs):
            s = match(kind, compose(kind, H[i: i + bs], qv[i: i + bs]), Ec)
            if not part.topic_answer:
                t = torch.as_tensor(topics[i: i + bs], device=device)
                hit = gid.unsqueeze(0) == t.unsqueeze(1)
                s = s.masked_fill(hit, float("-inf"))
            v, a = s.max(1)
            upd = v > best[i: i + bs]
            best[i: i + bs] = torch.where(upd, v, best[i: i + bs])
            best_id[i: i + bs] = torch.where(upd, gid[a], best_id[i: i + bs])
    return best_id.cpu().numpy()


def score(qs, top):
    hits = np.array([float(int(t) in set(q["answers"])) for q, t in zip(qs, top)])
    cross = np.array([bool(q.get("cross_client")) for q in qs])
    m = {"n": len(qs), "hits1": float(hits.mean()) if len(qs) else 0.0, "n_cross": int(cross.sum())}
    if cross.any():
        m["hits1_cross"] = float(hits[cross].mean())
    if (~cross).any():
        m["hits1_local"] = float(hits[~cross].mean())
    return m, hits


def fmt(m):
    s = f"Hits@1 {m['hits1']:.4f}"
    if "hits1_cross" in m:
        s += f" | cross {m['hits1_cross']:.4f}"
    if "hits1_local" in m:
        s += f" local {m['hits1_local']:.4f}"
    return s


# ----------------------------------------------------------------------------- FL-KG-QA (per client, KEQA)

@torch.no_grad()
def local_answer(model, qs, ids, E, g2l, view, args, device, pad_id, topic_answer):
    """Nearest entity to e_topic + r_q among the owner's stored entities (TransE distance)."""
    if not qs:
        return np.zeros(0, dtype=np.int64)
    qv = encode_questions(model, ids, args, device, pad_id)
    t = torch.as_tensor(g2l[np.array([q["topic_id"] for q in qs])], device=device)
    out = []
    for i in range(0, len(qs), args.eval_batch_size):
        tb = t[i: i + args.eval_batch_size]
        s = match("transe", E[tb] + qv[i: i + args.eval_batch_size], E)
        if not topic_answer:
            s[torch.arange(len(tb), device=device), tb] = float("-inf")
        out.append(view[s.argmax(1)].cpu().numpy())
    return np.concatenate(out)


def run_flkgqa(model, part, tables, R, train, dev, test, tr_ids, dv_ids, te_ids, args, device, pad_id, rng):
    init = model.trainable_state()
    top = np.full(len(test), -1, dtype=np.int64)
    history, n_tr = [], []
    epochs = args.qa_rounds * args.qa_local_epochs
    for c in range(part.n_clients):
        E, g2l = tables[c], part.g2l[c]
        view = torch.as_tensor(part.view[c], device=device)
        ex = []
        for q, x in zip(train, tr_ids):
            lp = q.get("local_paths") or []
            if part.owner[q["topic_id"]] != c or not lp or q.get("local_f1", 0.0) < args.local_full_f1:
                continue
            tgt = torch.stack([R[torch.as_tensor(p, device=device)].sum(0) for p in lp]).mean(0)
            ex.append({"ids": x, "tgt": tgt})
        n_tr.append(len(ex))
        dv = [i for i, q in enumerate(dev) if part.owner[q["topic_id"]] == c]
        te = [i for i, q in enumerate(test) if part.owner[q["topic_id"]] == c]
        model.load_trainable_state(init)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        steps = max(1, math.ceil(len(ex) / args.batch_size) * epochs)
        warm = max(1, int(0.06 * steps))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / steps))
        best, best_state = -1.0, init
        for ep in range(epochs if ex else 0):
            ts = time.time()
            model.train()
            order = list(range(len(ex)))
            rng.shuffle(order)
            tot = 0.0
            for i in range(0, len(order), args.batch_size):
                b = [ex[j] for j in order[i: i + args.batch_size]]
                ids, att = pad_batch([e["ids"] for e in b], pad_id, device)
                with autocast(device):
                    qv = model(ids, att)
                loss = ((qv.float() - torch.stack([e["tgt"] for e in b])) ** 2).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                tot += loss.item() * len(b)
            dm, _ = score([dev[i] for i in dv], local_answer(model, [dev[i] for i in dv], [dv_ids[i] for i in dv], E, g2l,
                                                             view, args, device, pad_id, part.topic_answer))
            history.append({"client": c, "epoch": ep + 1, "loss": tot / max(1, len(ex)), "dev": dm})
            print(f"[client {c} epoch {ep + 1}/{epochs}] loss {tot / max(1, len(ex)):.4f} | dev {fmt(dm)} "
                  f"({time.time() - ts:.0f}s)")
            if dm["hits1"] > best or not dv:
                best, best_state = dm["hits1"], model.trainable_state()
        model.load_trainable_state(best_state)
        top[te] = local_answer(model, [test[i] for i in te], [te_ids[i] for i in te], E, g2l, view, args, device,
                               pad_id, part.topic_answer)
    print(f"[qa] FL-KG-QA training questions with a full local label: {n_tr}")
    return top, history, n_tr


# ----------------------------------------------------------------------------- KGE phase

def make_trainer(n_ent, n_rel, h, r, t, args, device, seed):
    full = args.kge_loss == "full" or (args.kge_loss == "auto" and n_ent <= args.kge_full_max)
    lr = args.kge_full_lr if full else args.kge_lr
    kind = "transe" if args.method == "flkgqa" else "complex"
    return KGETrainer(n_ent, n_rel, h, r, t, args.rank, lr, device, seed, full=full, kind=kind)


def kge_epochs(tr, n, args):
    return tr.train_epochs(n, args.kge_full_batch if tr.full else args.kge_batch, args.kge_neg, args.kge_reg)


def client_weights(part, device):
    """PFedEG* client-wise relation graph: W_ij = |E_i & E_j| / |E_i | E_j| (i != j), W_ii = min_j W_ij,
    rows normalised (Zhang et al., 2024)."""
    C = part.n_clients
    W = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if i != j:
                inter = np.intersect1d(part.view[i], part.view[j], assume_unique=True).size
                W[i, j] = inter / max(1, part.view[i].size + part.view[j].size - inter)
        W[i, i] = min(W[i, j] for j in range(C) if j != i) if C > 1 else 1.0
    W /= W.sum(1, keepdims=True)
    return torch.as_tensor(W, dtype=torch.float32, device=device)


@torch.no_grad()
def personalize(tables, views, n_ent, W, p):
    """PFedEG personalized supplementary embeddings: K_c = p * Norm(sum_j W_cj G_j) + (1 - p) * E_c."""
    out = []
    d = tables[0].shape[1]
    for c in range(len(tables)):
        num = torch.zeros((n_ent, d), device=tables[0].device)
        den = torch.zeros(n_ent, device=tables[0].device)
        for j, T in enumerate(tables):
            num.index_add_(0, views[j], W[c, j] * T)
            den.index_add_(0, views[j], W[c, j].expand(views[j].numel()).contiguous())
        K = num[views[c]] / den[views[c]].clamp(min=1e-12).unsqueeze(1)
        out.append(p * K + (1 - p) * tables[c])
        del num, den
    return out


def train_kge(part, args, device, log):
    rank = args.rank
    if args.method == "embedkgqa":
        tri = np.concatenate(part.triples)
        tr = make_trainer(part.n_ent, part.n_rel, tri[:, 0], tri[:, 1], tri[:, 2], args, device, args.seed)
        print(f"[kge] centralized ComplEx, {'full' if tr.full else 'sampled'} softmax, {len(tri)} triples")
        every = max(1, args.kge_epochs // 5)
        for ep in range(0, args.kge_epochs, every):
            n = min(every, args.kge_epochs - ep)
            ts = time.time()
            loss = kge_epochs(tr, n, args)
            mrr, h10 = tr.tail_mrr()
            log.append({"epoch": ep + n, "loss": loss, "mrr": mrr, "hits10": h10})
            print(f"[kge] epoch {ep + n}/{args.kge_epochs} loss {loss:.4f} | train-triple MRR {mrr:.3f} "
                  f"H@10 {h10:.3f} ({time.time() - ts:.0f}s)")
        return [tr.model.ent.weight.detach()], tr.model.rel.weight.detach()
    trainers = []
    held = np.bincount(np.concatenate(part.view), minlength=part.n_ent)   # clients that store each entity
    if args.method == "pflu":
        # global anchor set (the server sees the overlap through private set intersection): the most
        # connected entities stored by two or more clients; every client uses the ones it stores
        tri = np.concatenate(part.triples)
        gdeg = np.bincount(np.concatenate([tri[:, 0], tri[:, 2]]), minlength=part.n_ent)
        shared = np.nonzero(held > 1)[0]
        g_anchor = np.zeros(part.n_ent, dtype=bool)
        g_anchor[shared[np.argsort(-gdeg[shared], kind="stable")[: int(np.ceil(args.pflu_ratio * part.n_ent))]]] = True
    for c in range(part.n_clients):
        arr, g2l = part.triples[c], part.g2l[c]
        if args.method == "pflu":
            n = part.view[c].size
            full = args.kge_loss == "full" or (args.kge_loss == "auto" and n <= args.kge_full_max)
            trainers.append(AnchorTrainer(n, part.n_rel, g2l[arr[:, 0]], arr[:, 1], g2l[arr[:, 2]], rank, args.pflu_lr,
                                          device, args.seed + c, full=full, ratio=1.0, k=args.pflu_k,
                                          m=args.pflu_m, candidates=g_anchor[part.view[c]]))
        else:
            trainers.append(make_trainer(part.view[c].size, part.n_rel, g2l[arr[:, 0]], arr[:, 1], g2l[arr[:, 2]], args,
                                         device, args.seed + c))
    print(f"[kge] {args.method}: {'full' if trainers[0].full else 'sampled'} softmax per client")
    # FedE/FedR start from a common relation table (and FedE from common entity vectors)
    relcount = torch.zeros((part.n_clients, part.n_rel), device=device)
    for c, tr in enumerate(trainers):
        relcount[c] = torch.bincount(tr.R, minlength=part.n_rel).float()
    views = [torch.as_tensor(v, device=device) for v in part.view]

    Wc = client_weights(part, device) if args.method == "pfedeg" else None
    if args.method == "pflu":
        a_gid = [views[c][torch.as_tensor(tr.anchors, device=device)] for c, tr in enumerate(trainers)]
        print(f"[kge] pflu anchors per client: {[int(a.numel()) for a in a_gid]}; shared by 2+ clients: "
              f"{int((torch.bincount(torch.cat(a_gid), minlength=part.n_ent) > 1).sum())}")

    def aggregate():
        with torch.no_grad():
            if args.method in ("fede", "fedr", "flkgqa"):       # PFedEG and PFLU keep relations local
                W = relcount / relcount.sum(0, keepdim=True).clamp(min=1)
                R = sum(W[c].unsqueeze(1) * tr.model.rel.weight for c, tr in enumerate(trainers))
                used = relcount.sum(0) > 0
                for tr in trainers:
                    tr.model.rel.weight[used] = R[used]
            if args.method == "pfedeg":
                K = personalize([tr.model.ent.weight for tr in trainers], views, part.n_ent, Wc, args.pfedeg_p)
                for tr, Kc in zip(trainers, K):
                    tr.model.ent.weight.copy_(Kc)
                    tr.prox, tr.beta = Kc.clone(), args.pfedeg_beta
            if args.method == "pflu":                           # average anchor embeddings over their holders
                S = torch.zeros((part.n_ent, 2 * rank), device=device)
                N = torch.zeros(part.n_ent, device=device)
                for g, tr in zip(a_gid, trainers):
                    S.index_add_(0, g, tr.model.anchor[:-1])
                    N.index_add_(0, g, torch.ones(g.numel(), device=device))
                S /= N.clamp(min=1).unsqueeze(1)
                for g, tr in zip(a_gid, trainers):
                    tr.model.anchor[:-1].copy_(S[g])
            if args.method == "fede":
                S = torch.zeros((part.n_ent, 2 * rank), device=device)
                N = torch.zeros(part.n_ent, device=device)
                for c, tr in enumerate(trainers):
                    S.index_add_(0, views[c], tr.model.ent.weight)
                    N.index_add_(0, views[c], torch.ones(views[c].numel(), device=device))
                S /= N.clamp(min=1).unsqueeze(1)
                for c, tr in enumerate(trainers):
                    tr.model.ent.weight.copy_(S[views[c]])
        for tr in trainers:
            tr.reset_optimizer()

    aggregate()
    for rd in range(args.kge_rounds):
        ts = time.time()
        losses = [kge_epochs(tr, args.kge_local_epochs, args) for tr in trainers]
        aggregate()
        if (rd + 1) % max(1, args.kge_rounds // 5) == 0 or rd + 1 == args.kge_rounds:
            fit = [tr.tail_mrr() for tr in trainers]
            log.append({"round": rd + 1, "loss": losses, "mrr": [f[0] for f in fit], "hits10": [f[1] for f in fit]})
            print(f"[kge] round {rd + 1}/{args.kge_rounds} loss {np.round(losses, 4).tolist()} | train-triple MRR "
                  f"{np.round([f[0] for f in fit], 3).tolist()} ({time.time() - ts:.0f}s)")
    if args.method == "pflu":
        return [tr.table() for tr in trainers], trainers[0].model.rel.weight.detach()
    return [tr.model.ent.weight.detach() for tr in trainers], trainers[0].model.rel.weight.detach()


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--method", required=True, choices=["embedkgqa", "fede", "fedr", "flkgqa", "pfedeg", "pflu"])
    ap.add_argument("--pfedeg-p", type=float, default=0.7, help="PFedEG residual mix p")
    ap.add_argument("--pfedeg-beta", type=float, default=5e-4, help="PFedEG proximal weight")
    ap.add_argument("--pflu-ratio", type=float, default=0.1, help="PFLU anchor share of each client's entities")
    ap.add_argument("--pflu-k", type=int, default=20)
    ap.add_argument("--pflu-m", type=int, default=15)
    ap.add_argument("--pflu-lr", type=float, default=5e-3)
    ap.add_argument("--k-max", type=int, default=3, help="unused (kept for script compatibility)")
    ap.add_argument("--local-full-f1", type=float, default=0.999,
                    help="FL-KG-QA: a client's own path is a label when its answer F1 reaches this (WebQSP: 0.5)")
    ap.add_argument("--encoder", default="hf", choices=["hf", "bow"])
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--rank", type=int, default=200, help="ComplEx rank (embedding size 2 x rank)")
    ap.add_argument("--kge-lr", type=float, default=1e-2)
    ap.add_argument("--kge-epochs", type=int, default=100, help="centralized KGE epochs")
    ap.add_argument("--kge-rounds", type=int, default=20, help="federated KGE rounds")
    ap.add_argument("--kge-local-epochs", type=int, default=5)
    ap.add_argument("--kge-batch", type=int, default=4096)
    ap.add_argument("--kge-neg", type=int, default=1024)
    ap.add_argument("--kge-reg", type=float, default=1e-3)
    ap.add_argument("--kge-loss", default="auto", choices=["auto", "full", "sampled"],
                    help="auto: full softmax when the trainer knows at most --kge-full-max entities")
    ap.add_argument("--kge-full-max", type=int, default=300_000)
    ap.add_argument("--kge-full-lr", type=float, default=0.1, help="Adagrad learning rate for full softmax")
    ap.add_argument("--kge-full-batch", type=int, default=1000)
    ap.add_argument("--tune-ent", type=int, default=1,
                    help="1: fine-tune entity embeddings while training the QA model (EmbedKGQA --freeze 0); "
                         "FedE re-averages shared entities after every QA round, FedR keeps them local")
    ap.add_argument("--ent-lr", type=float, default=5e-4)
    ap.add_argument("--ent-drop", type=float, default=0.1)
    ap.add_argument("--rel-drop", type=float, default=0.2)
    ap.add_argument("--qa-rounds", type=int, default=10, help="QA rounds (centralized: epochs = rounds x local)")
    ap.add_argument("--qa-local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dev-size", type=int, default=2000)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baselines")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    part_name = os.path.basename(os.path.normpath(args.data_dir))
    hops = os.path.basename(os.path.dirname(os.path.normpath(args.data_dir)))
    tag = args.tag or f"{hops}_{part_name}_{args.method}_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    part = Partition(args.data_dir)
    train = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.max_train)
    dev = read_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    if args.dev_size and len(dev) > args.dev_size:
        dev = random.Random(args.seed + 1).sample(dev, args.dev_size)
    print(f"[data] {part_name} ({hops}): {len(train)} train, {len(dev)} dev, {len(test)} test; {part.n_clients} clients, "
          f"{part.n_ent} entities, {part.n_rel} relations; method {args.method}; device {device}")
    print(f"[views] entities stored per client: {[int(v.size) for v in part.view]}")

    kge_log = []
    tables, rel_table = train_kge(part, args, device, kge_log)
    central = args.method == "embedkgqa"
    kind = "transe" if args.method == "flkgqa" else "complex"

    tok = Tok(args, [q["question_masked"] for q in train])
    tr_ids, dv_ids, te_ids = tok.encode(train), tok.encode(dev), tok.encode(test)
    if args.encoder == "bow":
        enc = BowEncoder(len(tok.tok.vocab), 256)
    else:
        enc = HFEncoder(args.model_name, args.lora_r, args.lora_alpha, 0.1)
    model = QAModel(enc, args.rank).to(device)
    if args.method == "flkgqa":
        top, history, n_tr = run_flkgqa(model, part, tables, rel_table, train, dev, test, tr_ids, dv_ids, te_ids,
                                        args, device, tok.pad_id, rng)
    else:
        top, history, n_tr = run_federated_qa(model, part, tables, central, kind, train, dev, test, tr_ids, dv_ids,
                                              te_ids, args, device, tok.pad_id, rng)
    tm, hits = score(test, top)
    print(f"[test] {fmt(tm)} (n={tm['n']}, cross {tm['n_cross']})")
    res = {"tag": tag, "method": args.method, "args": vars(args), "data": part_name, "hops": hops,
           "kge": kge_log, "qa_train_questions": n_tr, "history": history, "test": tm,
           "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "test_preds.jsonl"), "w") as f:
        for q, t, h in zip(test, top, hits):
            f.write(json.dumps({"qid": q["qid"], "top": int(t), "hit": float(h), "cross": bool(q.get("cross_client"))}) + "\n")
    print(f"[done] {out_dir}/metrics.json in {(time.time() - t0) / 60:.1f} min")


def run_federated_qa(model, part, tables, central, kind, train, dev, test, tr_ids, dv_ids, te_ids, args, device,
                     pad_id, rng):

    if central:
        client_ex = [make_examples(train, tr_ids, np.arange(part.n_ent), part.topic_answer)]
    else:
        client_ex = []
        for c in range(part.n_clients):
            idx = [i for i, q in enumerate(train) if part.owner[q["topic_id"]] == c]
            client_ex.append(make_examples([train[i] for i in idx], [tr_ids[i] for i in idx], part.g2l[c],
                                           part.topic_answer))
    n_tr = [len(e) for e in client_ex]
    print(f"[qa] training questions with a known answer: {n_tr} of {len(train)}")

    tables = [nn.Parameter(T.clone(), requires_grad=bool(args.tune_ent)) for T in tables]
    views = [torch.as_tensor(v, device=device) for v in part.view]
    Wc = client_weights(part, device) if args.method == "pfedeg" else None

    def average_shared_entities():
        """FedE: average each entity's embedding over the clients that hold it."""
        with torch.no_grad():
            S = torch.zeros((part.n_ent, tables[0].shape[1]), device=device)
            N = torch.zeros(part.n_ent, device=device)
            for c, T in enumerate(tables):
                S.index_add_(0, views[c], T)
                N.index_add_(0, views[c], torch.ones(views[c].numel(), device=device))
            S /= N.clamp(min=1).unsqueeze(1)
            for c, T in enumerate(tables):
                T.copy_(S[views[c]])

    glob = model.trainable_state()
    best, best_state, history = -1.0, glob, []
    best_tables = [T.detach().clone() for T in tables] if args.tune_ent else None
    for rd in range(args.qa_rounds):
        ts = time.time()
        states, weights, losses = [], [], []
        for c, ex in enumerate(client_ex):
            if not ex:
                continue
            model.load_trainable_state(glob)
            E = tables[0] if central else tables[c]
            losses.append(train_qa(model, ex, E, args.qa_local_epochs, args, device, pad_id, rng, part.topic_answer,
                                   kind))
            states.append(model.trainable_state())
            weights.append(len(ex))
        glob = fedavg(states, weights)
        model.load_trainable_state(glob)
        if args.tune_ent and args.method == "fede":
            average_shared_entities()
        if args.tune_ent and args.method == "pfedeg":
            with torch.no_grad():
                for T, K in zip(tables, personalize(tables, views, part.n_ent, Wc, args.pfedeg_p)):
                    T.copy_(K)
        dm, _ = score(dev, answer(model, dev, dv_ids, tables, part, central, args, device, pad_id, kind))
        history.append({"round": rd + 1, "loss": losses, "dev": dm})
        print(f"[qa round {rd + 1}/{args.qa_rounds}] loss {np.round(losses, 4).tolist()} | dev {fmt(dm)} "
              f"({time.time() - ts:.0f}s)")
        if dm["hits1"] > best:
            best, best_state = dm["hits1"], {k: v.clone() for k, v in glob.items()}
            if args.tune_ent:
                best_tables = [T.detach().clone() for T in tables]
    model.load_trainable_state(best_state)
    if args.tune_ent:
        with torch.no_grad():
            for T, B in zip(tables, best_tables):
                T.copy_(B)
    top = answer(model, test, te_ids, tables, part, central, args, device, pad_id, kind)
    return top, history, n_tr


if __name__ == "__main__":
    main()
