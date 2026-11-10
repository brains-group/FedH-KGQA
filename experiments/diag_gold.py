"""Diagnostic: how well can the best relation path answer each question, under three execution rules?

  set          plain set expansion (the current executor)
  no_return    the topic entity is removed from every frontier after the first hop
  simple       path-level: no entity may repeat along a path (checked for the best no_return sequence)

For each rule it reports the mean best answer-set F1 and the oracle Hits@1 with random tie-breaking
(the precision of the best path's endpoint set), which bounds what a perfect relation predictor can reach.

  python experiments/diag_gold.py --data-dir data/processed/metaqa/3hop/C3_rho0.5_balanced_s0 --n 2000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedh.data.graph import RelGraph, enumerate_paths  # noqa: E402
from fedh.fed.execute import FederatedKG  # noqa: E402


def best(paths, e0, answers, ta=False):
    ans = set(answers)
    top = (0.0, 0.0, None)
    for seq, ends in paths.items():
        ends = set(ends.tolist()) - (set() if ta else {e0})
        if not ends:
            continue
        hit = len(ends & ans)
        if hit == 0:
            continue
        p, r = hit / len(ends), hit / len(ans)
        f1 = 2 * p * r / (p + r)
        if (f1, p) > top[:2]:
            top = (f1, p, seq)
    return top


def simple_endpoints(g, e0, seq, cap=200_000):
    states = [(e0, (e0,))]
    for r in seq:
        nxt = []
        for e, vis in states:
            for t in g.expand(np.array([e], np.int64), int(r)).tolist():
                if t not in vis:
                    nxt.append((t, vis + (t,)))
                    if len(nxt) > cap:
                        return None
        states = nxt
    return set(e for e, _ in states)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="", help="also write the summary here")
    a = ap.parse_args()
    kg = FederatedKG(a.data_dir)
    H = np.concatenate([g.heads for g in kg.graphs]); R = np.concatenate([g.rels for g in kg.graphs])
    T = np.concatenate([g.tails for g in kg.graphs])
    g = RelGraph(kg.n_ent, H, R, T)
    qs = [json.loads(l) for l in open(os.path.join(a.data_dir, f"{a.split}.jsonl"))]
    rng = np.random.default_rng(a.seed)
    if a.n and len(qs) > a.n:
        qs = [qs[i] for i in rng.permutation(len(qs))[: a.n]]
    t0 = time.time()
    res = {"set": [], "no_return": [], "simple": []}
    for q in qs:
        e0, K, ans = q["topic_id"], q["K"], q["answers"]
        for rule, nr in (("set", False), ("no_return", True)):
            f1, p, seq = best(enumerate_paths(g, e0, K, no_return=nr), e0, ans, kg.topic_answer)
            res[rule].append((f1, p))
            if rule == "no_return":
                if seq is None:
                    res["simple"].append((0.0, 0.0))
                    continue
                ends = simple_endpoints(g, e0, seq)
                if ends is None:
                    res["simple"].append((f1, p))
                    continue
                ends -= set() if kg.topic_answer else {e0}
                hit = len(ends & set(ans))
                if not ends or not hit:
                    res["simple"].append((0.0, 0.0))
                else:
                    pp, rr = hit / len(ends), hit / len(ans)
                    res["simple"].append((2 * pp * rr / (pp + rr), pp))
    print(f"{os.path.normpath(a.data_dir)} ({a.split}, {len(qs)} questions, {time.time() - t0:.0f}s)")
    summary = {}
    for rule, v in res.items():
        v = np.array(v)
        summary[rule] = {"f1": float(v[:, 0].mean()), "f1_is_1": float(np.mean(v[:, 0] >= 0.999)),
                         "oracle_hits1": float(v[:, 1].mean())}
        print(f"  {rule:10s} best-path F1 mean {v[:, 0].mean():.3f}, F1=1 for {np.mean(v[:, 0] >= 0.999):.1%}, "
              f"oracle Hits@1 (precision) {v[:, 1].mean():.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"data_dir": a.data_dir, "split": a.split, "n": len(qs), "rules": summary}, f, indent=2)


if __name__ == "__main__":
    main()
