"""P1 data pipeline for MetaQA (design revision 4).

Builds one federated split of MetaQA:
  1. KB with inverse triples (each fact stored at owner(h) and its inverse at owner(t)).
  2. Gold relation-path labels from the pooled KG (cached, partition-independent).
  3. Entity ownership with a controlled cross-client ratio rho.
  4. Per question: owner, cross-client flag, masked question, and client-side
     labels: relation paths the owner can find in its own view, plus first-hop
     candidates for questions that leave the owner.

Example:
  python experiments/p1_prepare_data.py --metaqa-dir data/raw/MetaQA \
      --hops 2 --clients 3 --rho 0.5 --seed 0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.data.graph import RelGraph, enumerate_paths, score_paths  # noqa: E402
from fedh.data.metaqa import load_kb, load_qa, qa_path  # noqa: E402
from fedh.data.partition import (bfs_assignment, calibrate_rho, community_assignment,  # noqa: E402
                                 is_cross_client, target_sizes)

_G = None
_OWNER = None
_K = None
_NR = False      # no_return: paths may not pass back through the topic entity
_TA = False      # topic_answer: the topic entity may itself be an answer (PathQuestion)
_MIXED = False   # mixed hop counts (WebQSP): each question takes the best path length up to _K
_MAXF = 200_000  # skip expansions whose frontier exceeds this


def _gold_worker(group):
    e0, qs = group
    ks = range(1, _K + 1) if _MIXED else [_K]
    paths = {k: enumerate_paths(_G, e0, k, no_return=_NR, max_frontier=_MAXF) for k in ks}
    out = []
    for q in qs:
        best = None
        for k in ks:       # shorter paths win ties
            f1, top = score_paths(paths[k], e0, np.asarray(q["answers"]), topic_answer=_TA)
            if best is None or f1 > best[0] + 1e-9:
                best = (f1, top, k)
        f1, top, k = best
        out.append((q["qid"], f1, [t[0] for t in top], [[t[1], t[2]] for t in top], k))
    return out


def _local_worker(group):
    e0, qs = group
    c = int(_OWNER[e0])
    # The owner does not know a question's hop count. With fixed-hop benchmarks (MetaQA k-hop,
    # PQ-kH) the hop count is a public property of the dataset; with mixed hops (WebQSP) the owner
    # searches every length up to K_max on its own view and keeps the best (shorter wins ties),
    # never the length found on the pooled KG.
    ks = list(range(1, _K + 1)) if _MIXED else sorted(set(int(q["K"]) for q in qs))
    paths = {k: enumerate_paths(_G, e0, k, _OWNER, c, no_return=_NR, max_frontier=_MAXF) for k in ks}
    # first-hop relations from e0 that reach at least one entity the owner cannot expand
    r1 = []
    src = np.array([e0], dtype=np.int64)
    for r in _G.out_rels(src):
        t = _G.expand(src, int(r))
        if t.size and np.any(_OWNER[t] != c):
            r1.append(int(r))
    out = []
    for q in qs:
        best = (0.0, [])
        for k in (ks if _MIXED else [int(q["K"])]):
            f1, top = score_paths(paths[k], e0, np.asarray(q["answers"]), topic_answer=_TA)
            if f1 > best[0] + 1e-9:
                best = (f1, top)
        f1, top = best
        out.append((q["qid"], f1, [t[0] for t in top], r1))
    return out


def _run_pool(fn, groups, workers):
    if workers <= 1:
        return [fn(g) for g in groups]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(fn, groups, chunksize=max(1, len(groups) // (workers * 8)))


def _groups(qs):
    by = defaultdict(list)
    for q in qs:
        by[q["topic_id"]].append(q)
    return list(by.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metaqa-dir", default="data/raw/MetaQA")
    ap.add_argument("--out", default="data/processed/metaqa")
    ap.add_argument("--hops", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--rho", type=float, default=0.5, help="target cross-client ratio")
    ap.add_argument("--balance", default="balanced", choices=["balanced", "dirichlet"])
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--calib-size", type=int, default=3000, help="train questions used to calibrate rho")
    ap.add_argument("--limit", type=int, default=0, help="max questions per split (0 = all); for smoke tests")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--no-return", action="store_true",
                    help="paths may not pass back through the topic entity (execution rule, stored in stats.json)")
    ap.add_argument("--topic-answer", action="store_true",
                    help="the topic entity may be an answer (PathQuestion); MetaQA excludes it")
    ap.add_argument("--mixed-hops", action="store_true",
                    help="questions have 1 to --hops hops; each takes its best gold path length (WebQSP)")
    ap.add_argument("--max-frontier", type=int, default=200_000, help="skip path expansions larger than this")
    ap.add_argument("--partition", default="auto", choices=["auto", "louvain", "bfs"],
                    help="base partition before shuffling; auto = louvain below 1M facts, else bfs")
    args = ap.parse_args()

    global _G, _OWNER, _K, _NR, _TA, _MIXED, _MAXF
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    kb = load_kb(os.path.join(args.metaqa_dir, "kb.txt"), add_inverse=True)
    _G = RelGraph(kb.n_ent, kb.heads, kb.rels, kb.tails)
    _K = args.hops
    _NR, _TA, _MIXED, _MAXF = args.no_return, args.topic_answer, args.mixed_hops, args.max_frontier
    rule = ("_nr" if _NR else "") + ("_ta" if _TA else "")
    lab_rule = rule + ("_mixed" if _MIXED else "")
    print(f"[kb] {kb.n_ent} entities, {kb.n_facts} facts, {kb.heads.size} stored triples with inverses, "
          f"{kb.n_rel} relations incl. inverses")

    base_dir = os.path.join(args.out, f"{args.hops}hop")
    label_dir = os.path.join(base_dir, "_gold_labels" + lab_rule)
    os.makedirs(label_dir, exist_ok=True)

    splits = {}
    for split in args.splits.split(","):
        qs, dropped = load_qa(qa_path(args.metaqa_dir, args.hops, split), kb, args.hops, split)
        if args.limit:
            qs = qs[: args.limit]
        cache = os.path.join(label_dir, f"{split}{'_lim' + str(args.limit) if args.limit else ''}.jsonl")
        if os.path.exists(cache):
            labels = {}
            with open(cache) as f:
                for line in f:
                    r = json.loads(line)
                    labels[r["qid"]] = r
            print(f"[gold] {split}: loaded cached labels")
        else:
            ts = time.time()
            res = _run_pool(_gold_worker, _groups(qs), args.workers)
            labels = {}
            with open(cache, "w") as f:
                for chunk in res:
                    for qid, f1, paths, pr, k in chunk:
                        labels[qid] = {"qid": qid, "gold_f1": f1, "gold_paths": paths, "gold_pr": pr, "K": k}
                        f.write(json.dumps(labels[qid]) + "\n")
            print(f"[gold] {split}: labelled {len(qs)} questions in {time.time() - ts:.0f}s")
        for q in qs:
            q.update({k: v for k, v in labels[q["qid"]].items() if k != "qid"})
        f1s = np.array([q["gold_f1"] for q in qs])
        print(f"[gold] {split}: {len(qs)} questions ({dropped} dropped, topic not in KB); "
              f"gold-path F1 mean {f1s.mean():.3f}, F1=1 for {np.mean(f1s >= 0.999):.1%}")
        splits[split] = qs

    # ---- ownership ----
    sizes = target_sizes(kb.n_ent, args.clients, args.balance, args.alpha, rng)
    if args.clients == 1:
        owner, frac, rho_cal, trace = np.zeros(kb.n_ent, dtype=np.int64), 0.0, 0.0, []
    else:
        ts = time.time()
        fwd = np.arange(kb.heads.size) < kb.n_facts
        method = args.partition if args.partition != "auto" else ("louvain" if kb.n_facts < 1_000_000 else "bfs")
        part_fn = community_assignment if method == "louvain" else bfs_assignment
        base = part_fn(kb.heads[fwd], kb.tails[fwd], kb.n_ent, sizes, args.seed)
        print(f"[partition] base partition: {method}")
        train = splits.get("train") or next(iter(splits.values()))
        pool = [q for q in train if q["gold_paths"]]
        idx = rng.permutation(len(pool))[: args.calib_size]
        calib = [pool[i] for i in idx]
        owner, frac, rho_cal, trace = calibrate_rho(_G, base, calib, args.rho, args.seed, no_return=_NR)
        print(f"[partition] communities + shuffle fraction {frac:.3f} -> rho {rho_cal:.3f} on "
              f"{len(calib)} calibration questions (target {args.rho}) in {time.time() - ts:.0f}s")
    _OWNER = owner

    tag = f"C{args.clients}_rho{args.rho:g}_{args.balance}_s{args.seed}" + rule
    if args.limit:
        tag += f"_lim{args.limit}"
    out_dir = os.path.join(base_dir, tag)
    os.makedirs(os.path.join(out_dir, "clients"), exist_ok=True)

    stats = {"tag": tag, "hops": args.hops, "clients": args.clients, "target_rho": args.rho,
             "balance": args.balance, "seed": args.seed, "shuffle_fraction": frac,
             "calibration_rho": rho_cal, "calibration_trace": trace,
             "exec_rules": {"no_return": _NR, "topic_answer": _TA},
             "n_entities": kb.n_ent, "n_facts": kb.n_facts, "n_relations_with_inverse": kb.n_rel,
             "splits": {}, "clients_info": []}

    for split, qs in splits.items():
        ts = time.time()
        res = _run_pool(_local_worker, _groups(qs), args.workers)
        loc = {qid: (f1, paths, r1) for chunk in res for qid, f1, paths, r1 in chunk}
        n_cross = 0
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for q in qs:
                cross = is_cross_client(_G, q, owner, _NR) if args.clients > 1 else False
                n_cross += cross
                lf1, lpaths, r1 = loc[q["qid"]]
                rec = dict(q)
                rec.update({"owner": int(owner[q["topic_id"]]), "cross_client": bool(cross),
                            "local_f1": lf1, "local_paths": lpaths,
                            # computed from the owner's view only; used when it has no local path
                            "r1_candidates": r1})
                f.write(json.dumps(rec) + "\n")
        per_client = np.bincount([int(owner[q["topic_id"]]) for q in qs], minlength=args.clients)
        n_lab = sum(1 for q in qs if q["gold_paths"])
        rho = n_cross / max(1, n_lab)
        stats["splits"][split] = {"n": len(qs), "n_with_gold_path": n_lab, "rho": rho,
                                  "questions_per_client": per_client.tolist()}
        print(f"[{split}] rho {rho:.3f} over {n_lab} questions with a gold path; "
              f"questions per client {per_client.tolist()} ({time.time() - ts:.0f}s)")

    with open(os.path.join(out_dir, "entities.tsv"), "w", encoding="utf-8") as f:
        f.write("id\tname\towner\n")
        for i, name in enumerate(kb.id2ent):
            f.write(f"{i}\t{name}\t{int(owner[i])}\n")
    with open(os.path.join(out_dir, "relations.tsv"), "w", encoding="utf-8") as f:
        f.write("id\tname\tinverse_id\n")
        for i, name in enumerate(kb.id2rel):
            f.write(f"{i}\t{name}\t{int(kb.inverse_of[i])}\n")
    head_owner = owner[kb.heads]
    for c in range(args.clients):
        m = head_owner == c
        h, r, t = kb.heads[m], kb.rels[m], kb.tails[m]
        np.savetxt(os.path.join(out_dir, "clients", f"client_{c}.tsv"), np.stack([h, r, t], 1),
                   fmt="%d", delimiter="\t", header="head\trel\ttail", comments="")
        np.save(os.path.join(out_dir, "clients", f"client_{c}.npy"), np.stack([h, r, t], 1))
        boundary = np.unique(t[owner[t] != c])
        stats["clients_info"].append({"client": c, "owned_entities": int(np.sum(owner == c)),
                                      "stored_triples": int(m.sum()),
                                      "boundary_entities": int(boundary.size)})
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] wrote {out_dir} in {time.time() - t0:.0f}s")
    for ci in stats["clients_info"]:
        print(f"  client {ci['client']}: {ci['owned_entities']} entities, {ci['stored_triples']} triples, "
              f"{ci['boundary_entities']} boundary entities")


if __name__ == "__main__":
    main()
