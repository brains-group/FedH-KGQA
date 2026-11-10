"""Answer-set F1 for the relation-space runs (FedH-KGQA, Centralized, Isolated, Isolated + relay).

No retraining. Each run directory already holds test_probs.npz (per-hop relation probabilities P and
hop-count probabilities pK for every test question) and metrics.json (data_dir, exec mode, pruning).
For every test question we take the most likely hop count k and the most likely relation at each of
the k hops, execute that relation path from the topic entity with the run's own execution mode
(relay over all clients, or the owner only for isolated runs), and use the endpoints as the predicted
answer set. This is the standard path-execution answer set used by path-based KGQA methods.

Reported per run, overall and on cross-client questions:
  hits1     Hits@1 recomputed from the saved probabilities (sanity check against metrics.json)
  f1        mean per-question F1 between the predicted and the gold answer set
  precision, recall   means of the per-question values
  all_found share of questions whose predicted set contains every gold answer

  python experiments/p6_answer_f1.py --dataset "MetaQA 2-hop" --runs <run dir> ... --out results/p6/f1_metaqa2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedh.fed.execute import FederatedKG, follow_path_fed, hits_at_1, sparsify  # noqa: E402


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def label(a):
    if a["setting"] == "centralized":
        return "Centralized"
    if a["setting"] == "isolated":
        return "Isolated + relay" if a.get("exec_mode") == "relay" else "Isolated"
    return "FedH-KGQA"


def prf(pred, gold):
    if not gold:  # no gold answer in the KG: scored as a miss, as Hits@1 in metrics.json does
        return 0.0, 0.0, 0.0, 0.0
    hit = len(pred & gold)
    p = hit / len(pred) if pred else 0.0
    r = hit / len(gold)
    f = 2 * p * r / (p + r) if hit else 0.0
    return p, r, f, float(hit == len(gold))


def summarise(rows):
    if not rows:
        return {}
    a = np.array(rows)
    return {"n": len(rows), "hits1": a[:, 0].mean(), "precision": a[:, 1].mean(), "recall": a[:, 2].mean(),
            "f1": a[:, 3].mean(), "all_found": a[:, 4].mean()}


def score_run(run, kgs):
    with open(os.path.join(run, "metrics.json")) as f:
        m = json.load(f)
    a = m["args"]
    data_dir = a["data_dir"]
    if data_dir not in kgs:
        kgs[data_dir] = (FederatedKG(data_dir), {q["qid"]: q for q in read_jsonl(os.path.join(data_dir, "test.jsonl"))})
    kg, test = kgs[data_dir]
    z = np.load(os.path.join(run, "test_probs.npz"))
    mode = a.get("exec_mode") or ("local" if a["setting"] == "isolated" else "relay")
    allr, cross = [], []
    for qid, P, pK in zip(z["qid"].tolist(), z["P"].astype(np.float64), z["pK"].astype(np.float64)):
        q = test[qid]
        gold = set(int(x) for x in q["answers"])
        # Hits@1 exactly as in training-time evaluation (sanity check)
        Ps, pKs = sparsify(P, pK, a.get("sparsify_rel", 0.0), a.get("sparsify_top", 0))
        ids, sc = kg.answer_scores(q["topic_id"], Ps, pKs, mode=mode, prune=a.get("prune", 1e-3), cap=a.get("cap", 0))
        h = hits_at_1(ids, sc, gold)
        # predicted answer set: endpoints of the argmax relation path
        k = int(pK.argmax()) + 1
        seq = [int(P[t].argmax()) for t in range(k)]
        pred = set(follow_path_fed(kg, q["topic_id"], seq, mode=mode).tolist())
        if not kg.topic_answer:
            pred.discard(int(q["topic_id"]))
        r = prf(pred, gold)
        if r is None:
            continue
        row = (h,) + r
        allr.append(row)
        if q.get("cross_client"):
            cross.append(row)
    return label(a), m["test"]["hits1"], summarise(allr), summarise(cross)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kgs, out = {}, []
    print(f"[{a.dataset}]")
    for run in a.runs:
        if not os.path.exists(os.path.join(run, "test_probs.npz")):
            print(f"  skip (no test_probs.npz): {run}")
            continue
        lab, h_saved, s, c = score_run(run, kgs)
        flag = "" if abs(s["hits1"] - h_saved) < 0.005 else f"  <-- differs from metrics.json ({h_saved:.4f})"
        print(f"  {lab:18s} n={s['n']:5d} Hits@1 {s['hits1']:.4f} F1 {s['f1']:.4f} P {s['precision']:.4f} "
              f"R {s['recall']:.4f} all {s['all_found']:.4f}{flag}")
        if c:
            print(f"  {'':18s} cross n={c['n']:5d} Hits@1 {c['hits1']:.4f} F1 {c['f1']:.4f} R {c['recall']:.4f} "
                  f"all {c['all_found']:.4f}")
        out.append({"method": lab, "run": run, "hits1_saved": h_saved, "all": s, "cross": c})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"dataset": a.dataset, "rows": out}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
