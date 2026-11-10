"""Paired bootstrap significance of FedH-KGQA against every other method, per dataset.

Each run directory holds test_preds.jsonl (qid, per-question Hits@1) and metrics.json. For every
compared method, questions are matched by qid and resampled with replacement (paired bootstrap,
--n resamples). Reported: Hits@1 of both on the shared questions, the difference (ours - other),
its 95% percentile interval, and the one-sided p-value P(difference <= 0).

When several runs give the same method label (reruns), the most recent one is used and every
choice is printed with its test Hits@1, so it can be checked against the main table.

  python experiments/p5_significance.py --dataset "MetaQA 3-hop" --ours results/p3v2/metaqa3/<fed run> \\
      --others results/baselines_v2/metaqa3/* results/p3v2/metaqa3/<central run> --out results/p5/sig_metaqa3.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

NAMES = {"embedkgqa": "EmbedKGQA", "fede": "FedE", "fedr": "FedR", "flkgqa": "FL-KG-QA", "pfedeg": "PFedEG",
         "pflu": "PFLU", "transfernet": "TransferNet", "fedngdb": "FedNGDB"}


def label(m):
    if "method" in m:
        return NAMES.get(m["method"], m["method"])
    a = m["args"]
    if a["setting"] == "centralized":
        return "Centralized (ours)"
    if a["setting"] == "isolated":
        return "Isolated training + relay execution" if a.get("exec_mode") == "relay" else "Isolated"
    return "FedH-KGQA"


def preds(run):
    out = {}
    with open(os.path.join(run, "test_preds.jsonl")) as f:
        for line in f:
            p = json.loads(line)
            out[p["qid"]] = float(p["hit"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ours = preds(a.ours)
    runs = {}
    for r in a.others:
        mp, pp = os.path.join(r, "metrics.json"), os.path.join(r, "test_preds.jsonl")
        if not (os.path.exists(mp) and os.path.exists(pp)) or os.path.abspath(r) == os.path.abspath(a.ours):
            continue
        with open(mp) as f:
            m = json.load(f)
        lab = label(m)
        t = os.path.getmtime(pp)
        if lab not in runs or t > runs[lab][0]:
            runs[lab] = (t, r, m["test"]["hits1"])
    rng = np.random.default_rng(a.seed)
    rows = []
    print(f"[{a.dataset}] ours: {a.ours} ({len(ours)} questions)")
    for lab, (_, r, h) in sorted(runs.items(), key=lambda kv: -kv[1][2]):
        other = preds(r)
        q = sorted(set(ours) & set(other))
        x = np.array([ours[k] for k in q])
        y = np.array([other[k] for k in q])
        d = x - y
        idx = rng.integers(0, len(q), size=(a.n, len(q)))
        bs = d[idx].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        p = float((bs <= 0).mean())
        rows.append({"method": lab, "run": r, "n": len(q), "ours": float(x.mean()), "other": float(y.mean()),
                     "diff": float(d.mean()), "ci95": [float(lo), float(hi)], "p_one_sided": p})
        print(f"  vs {lab:36s} n={len(q):5d} ours {x.mean():.4f} other {y.mean():.4f} | diff {100 * d.mean():+.2f} pts "
              f"[{100 * lo:+.2f}, {100 * hi:+.2f}] p={p:.4f} | {r}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"dataset": a.dataset, "ours": a.ours, "rows": rows, "resamples": a.n}, f, indent=2)


if __name__ == "__main__":
    main()
