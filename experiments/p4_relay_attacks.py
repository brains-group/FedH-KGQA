"""P4 attacks on the inference relay (G2, G3), plus its per-query cost.

Uses the test-set relation and hop-count probabilities of a trained FedH-KGQA model
(test_probs.npz) and replays every evaluation question through the pull relay (fedh/fed/sketch_relay.py)
in two forms:
  ours      OPRF-style tokens, count-min sketches (d x B), L entries per client per round with
            dummies drawn from the real score distribution, K_max rounds always, k answers per client
  naive     the same relay without padding: each client sends only its real entries, rounds stop
            when the predicted hop mass is used up, each client returns up to k real answers

1. Entity existence. An attacker (the server, or a client asking about an entity outside its own
   KG) wants to know whether entity e is in the frontier of a round. Positives: entities submitted
   in that round, not stored by the attacking client; negatives: as many random entities that were
   not, also not stored by it. Scores, on the summed sketch of our relay:
     raw IDs          the relay carries entity IDs: the attacker reads the frontier (AUC 1 by construction)
     public hash      tokens are an unkeyed hash of the entity, so anyone computes them and reads
                      the count-min estimate (server and clients)
     shared key       tokens are a keyed hash with a key every client holds: a client computes the
                      token of any entity (the server cannot, AUC as without a token)
     OPRF tokens      tokens come from an oblivious PRF whose key only the querier holds: neither
                      the server nor a client can compute the token of an entity it does not store;
                      its best guess is a token under a key of its own
2. Traffic analysis by the server. From what it observes per query (per-client message sizes per
   round, number of rounds, number of answer messages per client, and the summed sketches:
   occupancy, mass, maximum and a histogram of bucket values per round) it predicts
     topic owner     which client owns the topic entity (C classes)
     answer owner    which client owns the gold answers (C classes, majority owner)
     cross-client    whether the question is cross-client (binary)
     frontier size   whether the total number of real frontier entries is above the median (binary)
     hop count       the gold hop count, on datasets with mixed hop counts (WebQSP)
   Attackers: multinomial logistic regression and k-nearest neighbours on standardised features,
   5-fold cross-validation over questions; reported: accuracy against the majority class, and AUC
   for binary tasks. The best of the two attackers is the headline.
3. Cost: bytes per query (uplink + downlink, all clients) and relay wall time per query.

  python experiments/p4_relay_attacks.py --data-dir data/processed/metaqa/3hop/C3_rho0.5_balanced_s0_nr \\
      --probs results/p3v2/metaqa3/<run>/test_probs.npz --out results/p4/relay/metaqa3 \\
      --sparsify-rel 0.1 --sparsify-top 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.fed.execute import FederatedKG, hits_at_1, sparsify  # noqa: E402
from fedh.fed.sketch_relay import KEY, SketchRelay  # noqa: E402
from fedh.relay.sketch import (PadConfig, build_sketch, entity_tokens, randomized_round,  # noqa: E402
                               read_back)

HIST_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, np.inf])
ID_BYTES, SCORE_BYTES, FIELD_BYTES, CT_BYTES = 8, 4, 4, 64   # entity id, score, field element, answer ciphertext

_R = None          # relay
_KG = None
_Q = _P = _PK = None
_A = None          # args


# ----------------------------------------------------------------------------- simulation

def sketch_feats(sk):
    v = sk[0]
    nz = v[v != 0]
    h = np.histogram(nz, HIST_BINS)[0] / max(1, nz.size)
    return [float((v != 0).sum()), float(sk.sum() / sk.shape[0]), float(nz.max()) if nz.size else 0.0] + h.tolist()


def simulate(i, padded):
    """One question through the pull relay. Returns per-round observations and the frontier."""
    r, kg, a = _R, _KG, _A
    q, P, pK = _Q[i], _P[i], _PK[i]
    e0 = int(q["topic_id"])
    n = kg.n_clients
    rng = np.random.default_rng([a.seed, i, int(padded)])
    nonce = int(rng.integers(0, 2 ** 62))
    t0 = time.perf_counter()
    if padded:
        sk = r._aggregate([(np.array([e0], np.int64), np.array([1.0]))], r.seed_pad, nonce, rng, None)
    else:
        sk = build_sketch(entity_tokens([e0], KEY), [1.0], nonce, r.d, r.B)
    rounds = []
    frontier = [None] * n
    acc = [dict() for _ in range(n)]
    for t in range(P.shape[0]):
        active = pK[t:].sum() >= r.prune
        if not active and not padded:
            break
        if active:
            cand = [r._pull_scores(c, sk, nonce, frontier[c], P[t]) for c in range(n)]
        else:
            cand = [(np.zeros(0, np.int64), np.zeros(0))] * n
        if kg.no_return:
            cand = [(ids[ids != e0], sc[ids != e0]) for ids, sc in cand]
        if padded:
            sent = []
            for ids, sc in cand:
                if ids.size > r.pad.max_real:
                    k = np.argsort(-sc, kind="stable")[: r.pad.max_real]
                    ids, sc = ids[k], sc[k]
                sent.append((ids, sc))
            sk = r._aggregate(cand, r.pad, nonce, rng, None)
            sizes = [r.pad.L] * n
        else:
            sent = cand
            sk = np.zeros((r.d, r.B))
            for ids, sc in cand:
                if ids.size:
                    sk += build_sketch(entity_tokens(ids, KEY), randomized_round(sc, r.gamma, rng), nonce, r.d, r.B)
            sizes = [int(ids.size) for ids, _ in cand]
        rounds.append({"sizes": sizes, "n_real": [int(ids.size) for ids, _ in sent], "feats": sketch_feats(sk),
                       "sent": sent, "sk": sk})
        frontier = cand
        if pK[t] >= r.prune:
            for c, (ids, sc) in enumerate(cand):
                d = acc[c]
                for e, s in zip(ids.tolist(), (sc * pK[t]).tolist()):
                    d[e] = d.get(e, 0.0) + s
        if not padded and sum(x.size for x, _ in cand) == 0:
            break
    n_ans = []
    ids_out, sc_out = [], []
    for c, d in enumerate(acc):
        if not kg.topic_answer:
            d.pop(e0, None)
        n_ans.append(r.k if padded else min(r.k, len(d)))
        if d:
            e = np.fromiter(d.keys(), np.int64, len(d))
            s = np.fromiter(d.values(), np.float64, len(d))
            top = np.argsort(-s, kind="stable")[: r.k]
            ids_out.append(e[top])
            sc_out.append(s[top])
    secs = time.perf_counter() - t0
    hit = hits_at_1(np.concatenate(ids_out), np.concatenate(sc_out), q["answers"]) if ids_out else 0.0
    return rounds, n_ans, nonce, secs, hit


def existence(rounds, nonce, rng):
    """Per attacking client: scores of positives and negatives with and without the right token."""
    r, kg, a = _R, _KG, _A
    out = {"with_token": ([], []), "without_token": ([], [])}
    wrong_key = KEY ^ int(rng.integers(1, 2 ** 62))
    for rd in rounds:
        front = np.unique(np.concatenate([ids for ids, _ in rd["sent"]])) if rd["sent"] else np.zeros(0, np.int64)
        if front.size == 0:
            continue
        for c in range(kg.n_clients):
            stored = r.nbr_ids[c]                                # own + boundary entities of client c
            pos = front[~np.isin(front, stored)]
            if pos.size == 0:
                continue
            if pos.size > a.exist_per_round:
                pos = rng.choice(pos, a.exist_per_round, replace=False)
            neg = rng.integers(0, kg.n_ent, pos.size * 3)
            neg = neg[~np.isin(neg, stored) & ~np.isin(neg, front)][: pos.size]
            if neg.size == 0:
                continue
            for name, key in (("with_token", KEY), ("without_token", wrong_key)):
                out[name][0].append(read_back(rd["sk"], entity_tokens(pos, key), nonce))
                out[name][1].append(read_back(rd["sk"], entity_tokens(neg, key), nonce))
    return {k: (np.concatenate(v[0]) if v[0] else np.zeros(0), np.concatenate(v[1]) if v[1] else np.zeros(0))
            for k, v in out.items()}


def worker(i):
    q = _Q[i]
    res = {}
    rng = np.random.default_rng([_A.seed, 7, i])
    for padded in (True, False):
        rounds, n_ans, nonce, secs, hit = simulate(i, padded)
        K = _P.shape[1]
        sizes = np.zeros((K, _KG.n_clients))
        feats = np.zeros((K, len(rounds[0]["feats"]) if rounds else 13))
        for t, rd in enumerate(rounds):
            sizes[t] = rd["sizes"]
            feats[t] = rd["feats"]
        obs = np.concatenate([[len(rounds)], sizes.ravel(), n_ans, feats.ravel()])
        n_real = sum(sum(rd["n_real"]) for rd in rounds)
        up = (sum(sum(rd["sizes"]) for rd in rounds) * (ID_BYTES + SCORE_BYTES) if not padded
              else (len(rounds) * _KG.n_clients + 1) * _R.d * _R.B * FIELD_BYTES)
        down = (sum(sum(rd["sizes"]) for rd in rounds) * (ID_BYTES + SCORE_BYTES) * _KG.n_clients if not padded
                else (len(rounds) + 1) * _KG.n_clients * _R.d * _R.B * FIELD_BYTES)
        ans = sum(n_ans) * (CT_BYTES if padded else ID_BYTES + SCORE_BYTES)
        res["ours" if padded else "naive"] = {"obs": obs, "n_real": n_real, "bytes": up + down + ans,
                                               "secs": secs, "hit": hit}
        if padded:
            res["exist"] = existence(rounds, nonce, rng)
    ans_own = _KG.owner[np.asarray(q["answers"], np.int64)] if q["answers"] else np.array([-1])
    res["labels"] = {"topic_owner": int(_KG.owner[q["topic_id"]]),
                     "answer_owner": int(np.bincount(ans_own[ans_own >= 0], minlength=_KG.n_clients).argmax())
                     if (ans_own >= 0).any() else -1,
                     "cross": int(bool(q.get("cross_client"))), "K": int(q["K"])}
    return res


# ----------------------------------------------------------------------------- attackers

def auc(pos, neg):
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    ranks = np.empty(allv.size)
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-9] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def logreg(Xtr, ytr, Xte, n_cls, l2=1e-2, iters=400, lr=0.5):
    W = np.zeros((Xtr.shape[1] + 1, n_cls))
    A = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
    Y = np.eye(n_cls)[ytr]
    for _ in range(iters):
        Z = A @ W
        Z -= Z.max(1, keepdims=True)
        Pr = np.exp(Z)
        Pr /= Pr.sum(1, keepdims=True)
        G = A.T @ (Pr - Y) / A.shape[0] + l2 * np.vstack([W[:-1], np.zeros((1, n_cls))])
        W -= lr * G
    Z = np.hstack([Xte, np.ones((Xte.shape[0], 1))]) @ W
    Z -= Z.max(1, keepdims=True)
    Pr = np.exp(Z)
    return Pr / Pr.sum(1, keepdims=True)


def knn(Xtr, ytr, Xte, n_cls, k=15):
    out = np.zeros((Xte.shape[0], n_cls))
    for s in range(0, Xte.shape[0], 128):
        D = ((Xte[s: s + 128, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
        nn = np.argsort(D, 1)[:, :k]
        for c in range(n_cls):
            out[s: s + 128, c] = (ytr[nn] == c).mean(1)
    return out


def attack(X, y, n_cls, seed, folds=5, null=True):
    """Cross-validated attack; with null=True also the same attack on shuffled labels (the chance
    level of this attacker on this many questions, selection over attackers included)."""
    keep = y >= 0
    X, y = X[keep], y[keep]
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    probs = {"logreg": np.zeros((len(y), n_cls)), "knn": np.zeros((len(y), n_cls))}
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        Xtr, Xte = standardise(X[tr], X[te])
        probs["logreg"][te] = logreg(Xtr, y[tr], Xte, n_cls)
        probs["knn"][te] = knn(Xtr, y[tr], Xte, n_cls)
    major = float(np.bincount(y, minlength=n_cls).max() / len(y))
    out = {"n": int(len(y)), "majority_acc": major}
    for name, pr in probs.items():
        out[f"{name}_acc"] = float((pr.argmax(1) == y).mean())
        if n_cls == 2:
            out[f"{name}_auc"] = auc(pr[y == 1, 1], pr[y == 0, 1])
    out["best_acc"] = max(out["logreg_acc"], out["knn_acc"])
    if n_cls == 2:
        out["best_auc"] = max(out["logreg_auc"], out["knn_auc"], 1 - min(out["logreg_auc"], out["knn_auc"]))
    if null:
        nl = attack(X, np.random.default_rng(seed + 1).permutation(y), n_cls, seed, folds, null=False)
        out["null_best_acc"] = nl["best_acc"]
        if n_cls == 2:
            out["null_best_auc"] = nl["best_auc"]
    return out


# ----------------------------------------------------------------------------- main

def main():
    global _R, _KG, _Q, _P, _PK, _A
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=0, help="sketch width (0: 65536, or 262144 above 500k entities)")
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=1024)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-dummy-frac", type=float, default=0.25)
    ap.add_argument("--k-answers", type=int, default=10)
    ap.add_argument("--prune", type=float, default=1e-3)
    ap.add_argument("--sparsify-rel", type=float, default=0.1)
    ap.add_argument("--sparsify-top", type=int, default=3)
    ap.add_argument("--n-questions", type=int, default=1000)
    ap.add_argument("--calib-questions", type=int, default=300)
    ap.add_argument("--exist-per-round", type=int, default=20, help="positives per attacking client per round")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _A = a
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    kg = FederatedKG(a.data_dir)
    _KG = kg
    B = a.B or (262144 if kg.n_ent > 500_000 else 65536)
    z = np.load(a.probs)
    pos = {q: i for i, q in enumerate(z["qid"].tolist())}
    qs, Ps, Ks = [], [], []
    with open(os.path.join(a.data_dir, "test.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            if q["qid"] in pos and q["answers"]:
                p, k = sparsify(z["P"][pos[q["qid"]]].astype(np.float64), z["pK"][pos[q["qid"]]].astype(np.float64),
                                a.sparsify_rel, a.sparsify_top)
                qs.append(q)
                Ps.append(p)
                Ks.append(k)
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(qs))
    n_cal = min(a.calib_questions, len(qs) // 4)
    cal, ev = order[:n_cal], order[n_cal: n_cal + a.n_questions]
    # dummy-score distribution from calibration questions (as in P2)
    cal_pad = PadConfig(L=a.L, mass_per_slot=1.0, min_dummy_frac=a.min_dummy_frac, fixed_mass=False)
    cal_relay = SketchRelay(kg, B, a.d, cal_pad, a.gamma, a.tau, a.k_answers, a.prune, "pull")
    sample = []
    for i in cal:
        st = {"n_real": [], "truncated": [], "scaled": [], "real_scores": []}
        cal_relay.run(qs[i]["topic_id"], Ps[i], Ks[i], np.random.default_rng([a.seed, 99, int(i)]), st)
        sample += st["real_scores"]
    emp = np.concatenate(sample) if sample else np.zeros(0)
    pad = PadConfig(L=a.L, mass_per_slot=1.0, min_dummy_frac=a.min_dummy_frac, dummy_scores="empirical",
                    empirical=emp, fixed_mass=False)
    _R = SketchRelay(kg, B, a.d, pad, a.gamma, a.tau, a.k_answers, a.prune, "pull")
    _Q, _P, _PK = [qs[i] for i in ev], np.stack([Ps[i] for i in ev]), np.stack([Ks[i] for i in ev])
    print(f"[p4] {os.path.basename(os.path.normpath(a.data_dir))}: {kg.n_clients} clients, {kg.n_ent} entities, "
          f"{len(_Q)} questions (+{n_cal} calibration), B={B} d={a.d} L={a.L} K_max={_P.shape[1]}")
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = pool.map(worker, range(len(_Q)), chunksize=8)
    else:
        res = [worker(i) for i in range(len(_Q))]
    print(f"[p4] relay replay done ({time.time() - t0:.0f}s)")

    out = {"args": vars(a), "n_questions": len(_Q), "n_clients": kg.n_clients, "B": B, "K_max": int(_P.shape[1])}
    # 1. entity existence
    ex = {}
    for name in ("with_token", "without_token"):
        p = np.concatenate([r["exist"][name][0] for r in res])
        n = np.concatenate([r["exist"][name][1] for r in res])
        ex[name] = {"auc": auc(p, n), "n_pos": int(p.size), "n_neg": int(n.size)}
    wt, wo = ex["with_token"]["auc"], ex["without_token"]["auc"]
    out["existence"] = {"pairs": ex["with_token"]["n_pos"],
                        "raw_ids": {"server": 1.0, "client": 1.0},
                        "public_hash": {"server": wt, "client": wt},
                        "shared_key": {"server": wo, "client": wt},
                        "oprf_tokens": {"server": wo, "client": wo}}
    print(f"[existence] {ex['with_token']['n_pos']} positive pairs | AUC with the token {wt:.3f}, without it {wo:.3f}"
          f" | raw IDs 1.000/1.000, public hash {wt:.3f}/{wt:.3f}, shared key {wo:.3f}/{wt:.3f}, "
          f"OPRF tokens {wo:.3f}/{wo:.3f} (server/client)")
    # 2. traffic analysis
    lab = {k: np.array([r["labels"][k] for r in res]) for k in res[0]["labels"]}
    nr = np.array([r["naive"]["n_real"] for r in res])
    big = (nr > np.median(nr)).astype(int)
    tasks = {"topic_owner": (lab["topic_owner"], kg.n_clients), "answer_owner": (lab["answer_owner"], kg.n_clients),
             "cross_client": (lab["cross"], 2), "frontier_size": (big, 2)}
    if len(np.unique(lab["K"])) > 1:
        Kv = lab["K"] - lab["K"].min()
        tasks["hop_count"] = (Kv, int(Kv.max()) + 1)
    out["traffic"] = {}
    for relay in ("naive", "ours"):
        X = np.stack([r[relay]["obs"] for r in res]).astype(np.float64)
        out["traffic"][relay] = {}
        for tname, (y, nc) in tasks.items():
            r_ = attack(X, y, nc, a.seed)
            if r_ is None:
                continue
            out["traffic"][relay][tname] = r_
            extra = f" | AUC {r_['best_auc']:.3f} (shuffled labels {r_['null_best_auc']:.3f})" if nc == 2 else ""
            print(f"[traffic {relay:5s}] {tname:13s} acc {r_['best_acc']:.3f} (majority {r_['majority_acc']:.3f}, "
                  f"shuffled labels {r_['null_best_acc']:.3f}){extra}")
    # 3. cost
    out["cost"] = {}
    for relay in ("naive", "ours"):
        b = np.array([r[relay]["bytes"] for r in res])
        s = np.array([r[relay]["secs"] for r in res])
        h = np.array([r[relay]["hit"] for r in res])
        out["cost"][relay] = {"bytes_mean": float(b.mean()), "bytes_p95": float(np.quantile(b, 0.95)),
                              "secs_mean": float(s.mean()), "secs_p95": float(np.quantile(s, 0.95)),
                              "hits1": float(h.mean())}
        print(f"[cost {relay:5s}] {b.mean() / 1e6:.3f} MB per query (p95 {np.quantile(b, 0.95) / 1e6:.3f}), "
              f"{s.mean() * 1e3:.0f} ms relay time (p95 {np.quantile(s, 0.95) * 1e3:.0f}), Hits@1 {h.mean():.4f}")
    out["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(a.out, "attacks.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {a.out}/attacks.json in {out['minutes']:.1f} min")


if __name__ == "__main__":
    main()
