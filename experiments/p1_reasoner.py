"""P1: relation-space reasoner on one federated MetaQA partition (design revision 4).

Settings
  centralized  one model, all training questions, gold relation paths from the pooled KG
  federated    FedAvg over clients; each client trains on the questions it owns
  isolated     each client trains alone and answers over its own KG only

Label sources (what a client trains on)
  gold    relation paths found on the pooled KG (only a central server has these)
  relay   the owner's own paths when they reach every answer; otherwise the owner
          discovers the path through the relay by executing candidate relation
          sequences (first hop from its own view, later hops over all relations)
          and keeping those whose answer F1 is best. With the exact relay this
          equals pooled execution, so the labels match `gold`; the number of relay
          queries spent on discovery is reported.
  local   only what the owner sees: its local paths when they reach any answer,
          otherwise its first-hop relations that leave the client (partial label)

Execution at test time
  relay   all clients expand what they own (exact relay, raw IDs; P2 swaps in sketches)
  local   only the question owner expands (Isolated)

Examples
  python experiments/p1_reasoner.py --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \
      --setting federated --labels relay
  python experiments/p1_reasoner.py --data-dir data/processed/metaqa/2hop/C1_rho0_balanced_s0 \
      --setting centralized --labels gold
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.fed.dp import DPStep, calibrate_sigma, epsilon, poisson_batch  # noqa: E402
from fedh.fed.execute import FederatedKG, hits_at_1, sparsify  # noqa: E402
from fedh.model.reasoner import (BowEncoder, BowTokenizer, HFEncoder, Reasoner,  # noqa: E402
                                 bow_relation_features, hf_relation_features, reasoner_loss)

_KG = None
_KMAX, _MIXED_HOPS = 3, False   # set in main(): K_max, and whether hop counts vary across questions


# ----------------------------------------------------------------------------- data

def read_jsonl(path, limit=0):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def label_paths(q, source, kg, full_f1=0.999):
    """Returns (paths, kind, relay_queries). kind: full | partial | none.
    A client's own path counts as a full label when its answer F1 on the client's view is at least
    full_f1 (0.999 by default; lower for datasets like WebQSP where even gold paths overshoot)."""
    if source == "gold":
        return (q["gold_paths"], "full", 0) if q["gold_paths"] else ([], "none", 0)
    lp, lf1 = q.get("local_paths") or [], q.get("local_f1", 0.0)
    if source == "relay":
        if lp and lf1 >= full_f1:
            return lp, "full", 0
        if not q["gold_paths"]:
            return [], "none", 0
        n_r1 = int(kg.out_rels(q["topic_id"]).size)
        # exhaustive search: every first hop the owner sees, then every relation at later hops; with mixed
        # hop counts (WebQSP) the owner tries every length up to K_max, otherwise the dataset's hop count
        ks = range(1, _KMAX + 1) if _MIXED_HOPS else [q["K"]]
        return q["gold_paths"], "full", sum(n_r1 * kg.n_rel ** (k - 1) for k in ks)
    if source == "local":
        if lp and lf1 > 0:
            return lp, ("full" if lf1 >= full_f1 else "partial"), 0
        if q.get("r1_candidates"):
            return [[r] for r in q["r1_candidates"]], "partial", 0
        return [], "none", 0
    raise ValueError(source)


class Encoderish:
    """Tokenizes masked questions for either encoder type."""

    def __init__(self, args, train_texts):
        self.kind = args.encoder
        self.max_len = args.max_len
        if self.kind == "bow":
            self.tok = BowTokenizer.build(train_texts)
            self.mask = "[ent]"
        else:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(args.model_name)
            self.mask = self.tok.mask_token or "[ENT]"

    def text(self, q):
        return q["question_masked"].replace("[ENT]", self.mask)

    def encode_all(self, qs):
        texts = [self.text(q) for q in qs]
        if self.kind == "bow":
            return [self.tok.encode(t, self.max_len) for t in texts]
        enc = self.tok(texts, truncation=True, max_length=self.max_len)
        return enc["input_ids"]

    @property
    def pad_id(self):
        return 0 if self.kind == "bow" else self.tok.pad_token_id


def make_examples(qs, ids, source, kg, full_f1=0.999):
    ex, stats = [], {"full": 0, "partial": 0, "none": 0, "relay_queries": 0, "n": len(qs)}
    for q, inp in zip(qs, ids):
        paths, kind, nq = label_paths(q, source, kg, full_f1)
        stats[kind] += 1
        stats["relay_queries"] += nq
        if kind == "none":
            continue
        # hop count: known from a full label's path length; unknown (-1) for partial labels
        k = len(paths[0]) - 1 if kind == "full" else -1
        ex.append({"ids": inp, "paths": paths, "k": k, "qid": q["qid"]})
    return ex, stats


def beam_paths(P, pK, B):
    """Top-B relation sequences over all lengths 1..K_max, scored by log p(K) + sum_t log p_t(r_t).
    The client does not know a question's hop count, so lengths compete under the hop-count head."""
    logP = np.log(np.maximum(P, 1e-12))
    logK = np.log(np.maximum(pK, 1e-12))
    cands = []
    beams = [((), 0.0)]
    for t in range(P.shape[0]):
        top = np.argsort(-logP[t])[:B]
        beams = sorted(((seq + (int(r),), lp + float(logP[t, r])) for seq, lp in beams for r in top),
                       key=lambda x: -x[1])[:B]
        cands += [(seq, lp + float(logK[t])) for seq, lp in beams]
    cands.sort(key=lambda x: -x[1])
    return [list(seq) for seq, _ in cands[:B]]


def _discover_worker(item):
    """Execute candidate sequences; keep those with the best answer F1, if it reaches min_f1."""
    e0, seqs, answers, min_f1 = item
    ans = set(int(a) for a in answers)
    scored = []
    for seq in seqs:
        P = np.zeros((len(seq), _KG.n_rel))
        P[np.arange(len(seq)), seq] = 1.0
        pK = np.zeros(len(seq))
        pK[-1] = 1.0
        ids, _ = _KG.answer_scores(e0, P, pK, mode="relay")
        ends = set(ids.tolist())
        hit = len(ends & ans)
        if hit:
            p, r = hit / len(ends), hit / len(ans)
            scored.append((2 * p * r / (p + r), seq))
    if not scored:
        return [], len(seqs)
    best = max(f for f, _ in scored)
    found = [seq for f, seq in scored if f >= best - 1e-9] if best >= min_f1 else []
    return found, len(seqs)


def discover_labels(model, pending, client_ex, args, device, pad, workers):
    """Model-guided label discovery: for questions without a full label, execute the model's top-B
    relation sequences through the relay and keep those that return exactly the known answers."""
    if not pending:
        return 0, 0
    qs = [q for q, _ in pending]
    P, PK = predict(model, qs, [i for _, i in pending], args, device, pad)
    items = [(q["topic_id"], beam_paths(P[j], PK[j], args.discover), q["answers"], args.discover_min_f1)
             for j, q in enumerate(qs)]
    if workers > 1 and len(items) > 200:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_discover_worker, items, chunksize=32)
    else:
        res = [_discover_worker(it) for it in items]
    by_qid = {e["qid"]: e for e in client_ex}
    n_found, n_queries, keep = 0, 0, []
    for (q, ids), (found, nq) in zip(pending, res):
        n_queries += nq
        if found:
            n_found += 1
            k = len(found[0]) - 1
            found = [f for f in found if len(f) == k + 1]
            if q["qid"] in by_qid:
                by_qid[q["qid"]].update({"paths": found, "k": k})
            else:
                client_ex.append({"ids": ids, "paths": found, "k": k, "qid": q["qid"]})
        else:
            keep.append((q, ids))
    pending[:] = keep
    return n_found, n_queries


def collate(batch, pad_id, k_max, device):
    L = max(len(b["ids"]) for b in batch)
    P = max(len(b["paths"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    paths = torch.full((len(batch), P, k_max), -1, dtype=torch.long)
    pmask = torch.zeros((len(batch), P), dtype=torch.long)
    k = torch.tensor([b["k"] for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        att[i, : len(b["ids"])] = 1
        for j, p in enumerate(b["paths"]):
            p = p[:k_max]
            paths[i, j, : len(p)] = torch.tensor(p)
            pmask[i, j] = 1
    return ids.to(device), att.to(device), paths.to(device), pmask.to(device), k.to(device)


# ----------------------------------------------------------------------------- model

_REL_FEATS = None


def build_model(args, tokz, n_rel, rel_names=None, device="cpu"):
    global _REL_FEATS
    if args.encoder == "bow":
        enc = BowEncoder(len(tokz.tok.vocab), args.bow_dim)
    else:
        dp = getattr(args, "dp_epsilon", 0) > 0
        enc = HFEncoder(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout,
                        attn_implementation="eager" if dp else None, mask4d=dp)
    feats = None
    if args.rel_head != "id":
        if _REL_FEATS is None:
            if args.rel_head == "text" and args.encoder == "hf":
                _REL_FEATS = torch.cat([hf_relation_features(args.model_name, rel_names, device),
                                        bow_relation_features(rel_names)], 1)
            else:
                _REL_FEATS = bow_relation_features(rel_names)
        feats = _REL_FEATS
    return Reasoner(enc, n_rel, args.k_max, rel_feats=feats)


def autocast(device):
    if device.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return torch.autocast("cpu", enabled=False)


def train_epochs_dp(model, examples, epochs, args, device, pad_id, dp):
    """DP-SGD: Poisson-sampled steps over the client's n_total training questions (unlabelled ones
    contribute nothing), per-example clipping, Gaussian noise; see fedh/fed/dp.py."""
    model.eval()                                   # dropout off: per-example gradients under vmap
    params = [p for p in model.parameters() if p.requires_grad]
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, round(1.0 / dp["q"]))
    steps = steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, 1 - s / steps)
        if args.schedule == "linear" else 1.0)
    # each of the n_total questions is drawn with probability q; only labelled ones are in `examples`
    q_ex = dp["q"]
    norms = []
    for _ in range(steps):
        idx = poisson_batch(len(examples), q_ex, dp["rng"])
        sums = None
        for i in range(0, idx.size, args.dp_microbatch):
            batch = [examples[j] for j in idx[i: i + args.dp_microbatch]]
            ids, att, paths, pmask, k = collate(batch, pad_id, args.k_max, device)
            g, nrm = dp["step"].clipped_sum(ids, att, paths, pmask, k)
            norms.append(nrm.mean().item())
            sums = g if sums is None else {n: sums[n] + g[n] for n in g}
        noisy = dp["step"].finish(sums, named)
        opt.zero_grad(set_to_none=True)
        for n, p in named.items():
            p.grad = noisy[n]
        opt.step()
        sched.step()
    return float(np.mean(norms)) if norms else float("nan")


def train_epochs(model, examples, epochs, args, device, pad_id, rng):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, math.ceil(len(examples) / args.batch_size) * epochs)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.06 * steps))) * max(0.0, 1 - s / steps)
        if args.schedule == "linear" else 1.0)
    tot, n = 0.0, 0
    for _ in range(epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            batch = [examples[j] for j in order[i: i + args.batch_size]]
            ids, att, paths, pmask, k = collate(batch, pad_id, args.k_max, device)
            with autocast(device):
                rl, kl = model(ids, att)
            loss, lp, lk = reasoner_loss(rl.float(), kl.float(), paths, pmask, k)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            tot += loss.item() * len(batch)
            n += len(batch)
    return tot / max(1, n)


@torch.no_grad()
def predict(model, qs, ids, args, device, pad_id):
    model.eval()
    Ps, Ks = [], []
    for i in range(0, len(qs), args.eval_batch_size):
        batch = [{"ids": x, "paths": [], "k": 0} for x in ids[i: i + args.eval_batch_size]]
        x, att, _, _, _ = collate(batch, pad_id, args.k_max, device)
        with autocast(device):
            rl, kl = model(x, att)
        Ps.append(torch.softmax(rl.float(), -1).cpu().numpy())
        Ks.append(torch.softmax(kl.float(), -1).cpu().numpy())
    return np.concatenate(Ps), np.concatenate(Ks)


def _exec_worker(item):
    e0, P, pK, answers, mode, prune, cap, sp_rel, sp_top = item
    P, pK = sparsify(P, pK, sp_rel, sp_top)
    ids, sc = _KG.answer_scores(e0, P, pK, mode=mode, prune=prune, cap=cap)
    if ids.size == 0:
        return 0.0, -1, 0
    top = int(ids[np.argmax(sc)])
    return hits_at_1(ids, sc, answers), top, int(ids.size)


def evaluate(model, qs, ids, args, device, pad_id, kg, mode, workers):
    P, K = predict(model, qs, ids, args, device, pad_id)
    items = [(q["topic_id"], P[i], K[i], q["answers"], mode, args.prune, args.cap, args.sparsify_rel, args.sparsify_top)
             for i, q in enumerate(qs)]
    if workers > 1 and len(items) > 500:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_exec_worker, items, chunksize=64)
    else:
        res = [_exec_worker(it) for it in items]
    hits = np.array([r[0] for r in res])
    cross = np.array([bool(q.get("cross_client")) for q in qs])
    # relation-path accuracy: argmax hop count and argmax relation per hop
    kk = K.argmax(1) + 1
    path_ok = []
    for i, q in enumerate(qs):
        pred = [int(P[i, t].argmax()) for t in range(kk[i])]
        path_ok.append(pred in [list(p) for p in q["gold_paths"]])
    path_ok = np.array(path_ok)
    m = {"n": len(qs), "hits1": float(hits.mean()) if len(qs) else 0.0,
         "path_acc": float(path_ok.mean()) if len(qs) else 0.0,
         "k_acc": float(np.mean(kk == np.array([q["K"] for q in qs]))) if len(qs) else 0.0,
         "n_cross": int(cross.sum())}
    if cross.any():
        m["hits1_cross"] = float(hits[cross].mean())
    if (~cross).any():
        m["hits1_local"] = float(hits[~cross].mean())
    preds = [{"qid": q["qid"], "hit": float(r[0]), "top": r[1], "n_cand": r[2], "cross": bool(c)}
             for q, r, c in zip(qs, res, cross)]
    return m, preds, P, K


def fedavg(states, weights):
    tot = float(sum(weights))
    return {k: sum(s[k].float() * (w / tot) for s, w in zip(states, weights)).to(states[0][k].dtype)
            for k in states[0]}


def fmt(m):
    s = f"Hits@1 {m['hits1']:.4f} path {m['path_acc']:.4f}"
    if "hits1_cross" in m:
        s += f" | cross {m['hits1_cross']:.4f} local {m.get('hits1_local', float('nan')):.4f}"
    return s


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--setting", default="federated", choices=["centralized", "federated", "isolated"])
    ap.add_argument("--labels", default=None, choices=["gold", "relay", "local"],
                    help="default: gold for centralized, relay for federated, local for isolated")
    ap.add_argument("--exec", dest="exec_mode", default=None, choices=["relay", "local"],
                    help="default: local for isolated, relay otherwise")
    ap.add_argument("--encoder", default="hf", choices=["hf", "bow"])
    ap.add_argument("--model-name", default="roberta-base")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--bow-dim", type=int, default=256)
    ap.add_argument("--k-max", type=int, default=3)
    ap.add_argument("--rel-head", default="id", choices=["id", "text", "words"],
                    help="id: one output per relation; text: score relations by their names (pretrained "
                         "embeddings + words); words: names as bag of words only")
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=5, help="federated rounds (centralized: epochs = rounds x local epochs)")
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--discover", type=int, default=0,
                    help="model-guided label discovery: relay-execute the top-B predicted paths of questions "
                         "without a full label and keep those that return exactly the answers (0 = off)")
    ap.add_argument("--discover-start", type=int, default=2, help="first round after which discovery runs")
    ap.add_argument("--discover-min-f1", type=float, default=0.999,
                    help="accept a discovered path when its answer F1 reaches this (exact match by default)")
    ap.add_argument("--local-full-f1", type=float, default=0.999,
                    help="a client's own path is a full label when its local answer F1 reaches this")
    ap.add_argument("--discover-every", type=int, default=2, help="run discovery every this many rounds")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--schedule", default="linear", choices=["linear", "constant"])
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--prune", type=float, default=1e-3, help="skip relations / hops below this probability")
    ap.add_argument("--cap", type=int, default=0, help="per-client frontier cap L (0 = none)")
    ap.add_argument("--sparsify-rel", type=float, default=0.0,
                    help="per hop, keep relations with p >= this fraction of the top one and renormalise (0 = off)")
    ap.add_argument("--sparsify-top", type=int, default=0, help="per hop, keep at most this many relations (0 = no limit)")
    ap.add_argument("--dev-size", type=int, default=2000, help="dev questions scored each round (0 = all)")
    ap.add_argument("--max-train", type=int, default=0, help="limit training questions (0 = all)")
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/p1")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dp-epsilon", type=float, default=0.0,
                    help="federated setting: train each client with DP-SGD to this (epsilon, delta) over all rounds "
                         "(QA-pair-level DP); 0 = no DP")
    ap.add_argument("--dp-delta", type=float, default=1e-5)
    ap.add_argument("--dp-clip", type=float, default=1.0)
    ap.add_argument("--dp-batch", type=int, default=256, help="expected batch size (sampling rate q = this / n)")
    ap.add_argument("--dp-microbatch", type=int, default=32, help="examples per vmap call (memory only)")
    ap.add_argument("--type-skew", type=int, default=0,
                    help="question-type skew: each question type (gold relation path) is kept for training at only "
                         "this many clients (0 = no skew); test and dev questions are unchanged")
    args = ap.parse_args()
    args.labels = args.labels or {"centralized": "gold", "federated": "relay", "isolated": "local"}[args.setting]
    args.exec_mode = args.exec_mode or ("local" if args.setting == "isolated" else "relay")

    global _KG
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    part = os.path.basename(os.path.normpath(args.data_dir))
    hops = os.path.basename(os.path.dirname(os.path.normpath(args.data_dir)))
    name = (args.model_name.split("/")[-1] if args.encoder == "hf" else "bow") + \
        ("" if args.rel_head == "id" else f"-{args.rel_head}")
    lab = args.labels + (f"+disc{args.discover}" if args.discover and args.setting == "federated" else "")
    tag = args.tag or f"{hops}_{part}_{args.setting}_{lab}_{args.exec_mode}_{name}_s{args.seed}"
    out_dir = os.path.join(args.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    global _KMAX, _MIXED_HOPS
    kg = FederatedKG(args.data_dir)
    _KG = kg
    train = read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.max_train)
    dev = read_jsonl(os.path.join(args.data_dir, "dev.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "test.jsonl"), args.max_test)
    if args.dev_size and len(dev) > args.dev_size:
        dev = random.Random(args.seed + 1).sample(dev, args.dev_size)
    _KMAX, _MIXED_HOPS = args.k_max, len({q["K"] for q in train}) > 1

    def type_ok(q, c):
        """Type skew: a question type is trained only at `type_skew` clients, chosen by a fixed hash."""
        if not args.type_skew or not q.get("gold_paths"):
            return True
        t = "|".join(map(str, q["gold_paths"][0]))
        h = int.from_bytes(t.encode(), "little") % 1_000_003
        return (c - h) % kg.n_clients < args.type_skew
    print(f"[data] {part} ({hops}): {len(train)} train, {len(dev)} dev (scored per round), {len(test)} test; "
          f"{kg.n_clients} clients, {kg.n_rel} relations; device {device}")
    print(f"[setup] setting={args.setting} labels={args.labels} exec={args.exec_mode} encoder={name} -> {out_dir}")

    tokz = Encoderish(args, [q["question_masked"] for q in train])
    tr_ids, dv_ids, te_ids = tokz.encode_all(train), tokz.encode_all(dev), tokz.encode_all(test)
    pad = tokz.pad_id

    n_clients = kg.n_clients
    owner_of = lambda q: int(q["owner"]) if "owner" in q else int(kg.owner[q["topic_id"]])  # noqa: E731
    label_stats = {}
    history = []
    epochs_total = args.rounds * args.local_epochs

    if args.setting == "centralized":
        ex, st = make_examples(train, tr_ids, args.labels, kg, args.local_full_f1)
        label_stats["all"] = st
        print(f"[labels] {st}")
        model = build_model(args, tokz, kg.n_rel, kg.rel_names, device).to(device)
        best, best_state = (-1.0, -1.0), None
        for ep in range(epochs_total):
            ts = time.time()
            loss = train_epochs(model, ex, 1, args, device, pad, rng)
            dm, *_ = evaluate(model, dev, dv_ids, args, device, pad, kg, args.exec_mode, args.workers)
            history.append({"epoch": ep + 1, "loss": loss, "dev": dm})
            print(f"[epoch {ep + 1}/{epochs_total}] loss {loss:.4f} | dev {fmt(dm)} ({time.time() - ts:.0f}s)")
            if (dm["hits1"], dm["path_acc"]) > best:
                best, best_state = (dm["hits1"], dm["path_acc"]), model.trainable_state()
        model.load_trainable_state(best_state)
        tm, preds, tP, tK = evaluate(model, test, te_ids, args, device, pad, kg, args.exec_mode, args.workers)
        test_q, saved_states = test, {"global": best_state}

    elif args.setting == "federated":
        per = [[i for i, q in enumerate(train) if owner_of(q) == c and type_ok(q, c)] for c in range(n_clients)]
        client_ex, pending = [], []
        for c in range(n_clients):
            ex, st = make_examples([train[i] for i in per[c]], [tr_ids[i] for i in per[c]], args.labels, kg, args.local_full_f1)
            client_ex.append(ex)
            pending.append([(train[i], tr_ids[i]) for i in per[c] if label_paths(train[i], args.labels, kg, args.local_full_f1)[1] != "full"]
                           if args.discover else [])
            st.update({"discovered": 0, "discover_queries": 0})
            label_stats[f"client_{c}"] = st
            print(f"[labels] client {c}: {st}")
        model = build_model(args, tokz, kg.n_rel, kg.rel_names, device).to(device)
        glob = model.trainable_state()
        best, best_state = (-1.0, -1.0), None
        dps = [None] * n_clients
        if args.dp_epsilon > 0:
            nprng = np.random.default_rng(args.seed + 7)
            for c in range(n_clients):
                n_tot = len(per[c])
                q = min(1.0, args.dp_batch / max(1, n_tot))
                T = max(1, round(1.0 / q)) * args.local_epochs * args.rounds
                sig = calibrate_sigma(q, T, args.dp_delta, args.dp_epsilon)
                dps[c] = {"q": q, "rng": nprng, "step": DPStep(model, args.dp_clip, sig, q * n_tot)}
                eps = epsilon(q, sig, T, args.dp_delta)
                label_stats[f"client_{c}"].update({"dp_sigma": sig, "dp_q": q, "dp_steps": T, "dp_epsilon": eps,
                                                   "dp_delta": args.dp_delta})
                print(f"[dp] client {c}: n={n_tot} q={q:.4f} steps={T} sigma={sig:.3f} -> epsilon {eps:.2f} "
                      f"(delta {args.dp_delta})")
        for rd in range(args.rounds):
            ts = time.time()
            states, weights, losses = [], [], []
            for c in range(n_clients):
                if not client_ex[c]:
                    continue
                model.load_trainable_state(glob)
                if dps[c] is not None:
                    losses.append(train_epochs_dp(model, client_ex[c], args.local_epochs, args, device, pad, dps[c]))
                else:
                    losses.append(train_epochs(model, client_ex[c], args.local_epochs, args, device, pad, rng))
                states.append(model.trainable_state())
                weights.append(len(per[c]) if dps[c] is not None else len(client_ex[c]))
            glob = fedavg(states, weights)
            model.load_trainable_state(glob)
            if args.discover and rd + 1 >= args.discover_start and (rd + 1 - args.discover_start) % args.discover_every == 0:
                for c in range(n_clients):
                    nf, nq = discover_labels(model, pending[c], client_ex[c], args, device, pad, args.workers)
                    label_stats[f"client_{c}"]["discovered"] += nf
                    label_stats[f"client_{c}"]["discover_queries"] += nq
                print("[discover] " + " | ".join(
                    f"client {c}: +{label_stats[f'client_{c}']['discovered']} labelled, "
                    f"{len(pending[c])} left, {label_stats[f'client_{c}']['discover_queries']} queries"
                    for c in range(n_clients)))
            dm, *_ = evaluate(model, dev, dv_ids, args, device, pad, kg, args.exec_mode, args.workers)
            history.append({"round": rd + 1, "client_loss": losses, "dev": dm})
            print(f"[round {rd + 1}/{args.rounds}] client loss {np.round(losses, 4).tolist()} | dev {fmt(dm)} "
                  f"({time.time() - ts:.0f}s)")
            if (dm["hits1"], dm["path_acc"]) > best:
                best, best_state = (dm["hits1"], dm["path_acc"]), {k: v.clone() for k, v in glob.items()}
        model.load_trainable_state(best_state)
        tm, preds, tP, tK = evaluate(model, test, te_ids, args, device, pad, kg, args.exec_mode, args.workers)
        test_q, saved_states = test, {"global": best_state}

    else:  # isolated
        preds, per_client = [], {}
        test_q, Ps, Kps, saved_states = [], [], [], {}
        for c in range(n_clients):
            idx = [i for i, q in enumerate(train) if owner_of(q) == c and type_ok(q, c)]
            ex, st = make_examples([train[i] for i in idx], [tr_ids[i] for i in idx], args.labels, kg, args.local_full_f1)
            label_stats[f"client_{c}"] = st
            print(f"[labels] client {c}: {st}")
            torch.manual_seed(args.seed)
            model = build_model(args, tokz, kg.n_rel, kg.rel_names, device).to(device)
            dv = [i for i, q in enumerate(dev) if owner_of(q) == c]
            dqs, dids = [dev[i] for i in dv], [dv_ids[i] for i in dv]
            best, best_state = (-1.0, -1.0), model.trainable_state()
            for ep in range(epochs_total):
                ts = time.time()
                loss = train_epochs(model, ex, 1, args, device, pad, rng) if ex else float("nan")
                dm, *_ = evaluate(model, dqs, dids, args, device, pad, kg, args.exec_mode, args.workers)
                history.append({"client": c, "epoch": ep + 1, "loss": loss, "dev": dm})
                print(f"[client {c} epoch {ep + 1}/{epochs_total}] loss {loss:.4f} | dev {fmt(dm)} "
                      f"({time.time() - ts:.0f}s)")
                if (dm["hits1"], dm["path_acc"]) > best:
                    best, best_state = (dm["hits1"], dm["path_acc"]), model.trainable_state()
            model.load_trainable_state(best_state)
            tv = [i for i, q in enumerate(test) if owner_of(q) == c]
            m, p, cP, cK = evaluate(model, [test[i] for i in tv], [te_ids[i] for i in tv], args, device, pad, kg,
                                    args.exec_mode, args.workers)
            per_client[c] = m
            preds += p
            test_q += [test[i] for i in tv]
            Ps.append(cP)
            Kps.append(cK)
            saved_states[f"client_{c}"] = best_state
        tP, tK = np.concatenate(Ps), np.concatenate(Kps)
        hits = np.array([p["hit"] for p in preds])
        cross = np.array([p["cross"] for p in preds])
        tm = {"n": len(preds), "hits1": float(hits.mean()),
              "path_acc": float(np.average([m["path_acc"] for m in per_client.values()],
                                           weights=[m["n"] for m in per_client.values()])),
              "n_cross": int(cross.sum()), "per_client": per_client}
        if cross.any():
            tm["hits1_cross"] = float(hits[cross].mean())
        if (~cross).any():
            tm["hits1_local"] = float(hits[~cross].mean())

    print(f"[test] {fmt(tm)} (n={tm['n']}, cross {tm['n_cross']})")
    q_relay = sum(s.get("relay_queries", 0) + s.get("discover_queries", 0) for s in label_stats.values())
    n_lab = sum(s.get("n", 0) for s in label_stats.values())
    if q_relay:
        print(f"[labels] relay discovery used {q_relay} path executions ({q_relay / max(1, n_lab):.1f} per training question)")
    res = {"tag": tag, "args": vars(args), "data": part, "hops": hops, "label_stats": label_stats,
           "history": history, "test": tm, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    # per-hop relation and hop-count probabilities on the test set (input to P2) and the trained weights
    np.savez_compressed(os.path.join(out_dir, "test_probs.npz"), qid=np.array([q["qid"] for q in test_q]),
                        P=tP.astype(np.float16), pK=tK.astype(np.float32))
    torch.save({k: {n: v.cpu() for n, v in st.items()} for k, st in saved_states.items()},
               os.path.join(out_dir, "trainable_state.pt"))
    with open(os.path.join(out_dir, "test_preds.jsonl"), "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"[done] {out_dir}/metrics.json in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
