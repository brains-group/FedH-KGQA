"""P2: the trained reasoner through the identifier-hiding relay (design revision 4).

Takes the per-hop relation and hop-count probabilities that P1 saved for the test
set (test_probs.npz) and answers every test question three ways:
  exact         raw IDs, no cap (the P1 number)
  exact capped  raw IDs, each client's frontier capped at L minus the dummy share
  sketch        OPRF-style tokens in d x B count-min sketches with padding to L
                entries, fixed mass M, quantisation 1/gamma and secure-sum
Pass: sketch Hits@1 at most 5 points below exact.

Two relay modes (push: senders submit reached candidates; pull: owners compute their
own entities' scores from the frontier and submit only those) and two padding rules
(fixed total mass M; dummies drawn from the real score distribution) are compared.

It also measures what the padding rule needs on real scores: how often a client's
real mass exceeds M (scaling), how often the cap truncates, the real per-entry
score distribution, and whether rounds with few and many real candidates can be
told apart from the summed sketch (bucket-value KS against a null).

The probabilities come from the masked question only, so the same file serves
partitions with any number of clients (questions are matched by qid).

Example:
  python experiments/p2_sketch_relay.py \
      --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 \
      --probs results/p1/<run>/test_probs.npz --out results/p2/C3_rho0.5
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.fed.execute import FederatedKG, hits_at_1, sparsify  # noqa: E402
from fedh.fed.sketch_relay import SketchRelay  # noqa: E402
from fedh.relay.sketch import PadConfig, ks_statistic  # noqa: E402

_KG = _RELAY = _QS = _P = _PK = None
_SEED = 0
_CAP = 0


def _exact_worker(i):
    q = _QS[i]
    ids, sc = _KG.answer_scores(q["topic_id"], _P[i], _PK[i], mode="relay", cap=_CAP)
    return hits_at_1(ids, sc, q["answers"])


def _sketch_worker(i):
    q = _QS[i]
    rng = np.random.default_rng([_SEED, i])
    st = {"n_real": [], "truncated": [], "scaled": [], "real_scores": []}
    ids, sc, _ = _RELAY.run(q["topic_id"], _P[i], _PK[i], rng, st)
    rs = np.concatenate(st["real_scores"]) if st["real_scores"] else np.zeros(0, np.float32)
    if rs.size > 200:
        rs = rng.choice(rs, 200, replace=False)
    n = np.array(st["n_real"])
    return (hits_at_1(ids, sc, q["answers"]), int((n > 0).sum()), int(np.sum(np.array(st["scaled"]) & (n > 0))),
            int(np.sum(np.array(st["truncated"]) > 0)), float(n[n > 0].mean()) if (n > 0).any() else 0.0, rs)


def _pool_map(fn, n, workers):
    if workers > 1 and n > 200:
        with mp.get_context("fork").Pool(workers) as pool:
            return pool.map(fn, range(n), chunksize=32)
    return [fn(i) for i in range(n)]


def load(args):
    kg = FederatedKG(args.data_dir)
    z = np.load(args.probs, allow_pickle=False)
    pos = {q: i for i, q in enumerate(z["qid"].tolist())}
    qs, P, pK = [], [], []
    with open(os.path.join(args.data_dir, f"{args.split}.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            if q["qid"] in pos and q["answers"]:
                qs.append(q)
                Pq, Kq = sparsify(z["P"][pos[q["qid"]]].astype(np.float64), z["pK"][pos[q["qid"]]].astype(np.float64),
                                  args.sparsify_rel, args.sparsify_top)
                P.append(Pq)
                pK.append(Kq)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(qs))
    n_cal = min(args.calib_questions, len(qs) // 4)     # never more than a quarter of the questions
    calib = order[:n_cal]
    rest = order[n_cal:]
    if args.n_questions:
        rest = rest[: args.n_questions]
    pick = lambda idx: ([qs[i] for i in idx], np.stack([P[i] for i in idx]), np.stack([pK[i] for i in idx]))  # noqa: E731
    return kg, pick(calib), pick(rest)


def traffic(relay, qs, P, pK, n, seed):
    """Summed-sketch bucket values for rounds with few vs many real candidates."""
    rows = []
    for i in range(min(n, len(qs))):
        rng = np.random.default_rng([seed, 10_000 + i])
        _, _, sks = relay.run(qs[i]["topic_id"], P[i], pK[i], rng, None, keep_sketch=True)
        for sk, n_real in sks:
            v = sk[sk != 0]
            rows.append((n_real, float(sk.sum()) / sk.shape[0], int((sk[0] != 0).sum()),
                         rng.choice(v, min(300, v.size), replace=False) if v.size else v))
    if len(rows) < 8:
        return {}
    nr = np.array([r[0] for r in rows])
    lo, hi = np.quantile(nr, 0.25), np.quantile(nr, 0.75)
    small = [r for r in rows if r[0] <= lo]
    large = [r for r in rows if r[0] >= hi]
    if not small or not large or lo == hi:
        return {"note": "real frontier sizes do not vary enough to split", "n_real_q25": float(lo),
                "n_real_q75": float(hi)}
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(large))
    la, lb = [large[i] for i in half[: len(half) // 2]], [large[i] for i in half[len(half) // 2:]]
    cat = lambda rs: np.concatenate([r[3] for r in rs]) if rs else np.zeros(0)  # noqa: E731
    return {"rounds": len(rows), "n_real_small_max": float(lo), "n_real_large_min": float(hi),
            "mass_small": float(np.mean([r[1] for r in small])), "mass_large": float(np.mean([r[1] for r in large])),
            "occupancy_small": float(np.mean([r[2] for r in small])),
            "occupancy_large": float(np.mean([r[2] for r in large])),
            "bucket_value_KS": ks_statistic(cat(small), cat(large)),
            "bucket_value_KS_null": ks_statistic(cat(la), cat(lb))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--probs", required=True, help="test_probs.npz written by p1_reasoner.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", default="65536,262144")
    ap.add_argument("--d", default="3")
    ap.add_argument("--L", default="128")
    ap.add_argument("--modes", default="push,pull", help="relay modes (see fedh/fed/sketch_relay.py)")
    ap.add_argument("--pads", default="fixed:beta,fixed:empirical,matched:empirical",
                    help="padding rule:dummy scores; fixed = total mass M, matched = dummies drawn from real scores")
    ap.add_argument("--mass", default="0.25,1.0", help="mass per slot for fixed padding (M = mass * L)")
    ap.add_argument("--gamma", default="1024", help="quantisation grid 1/gamma (0 = none)")
    ap.add_argument("--min-dummy-frac", type=float, default=0.25)
    ap.add_argument("--tau", default="0.05", help="read-back threshold(s), comma-separated")
    ap.add_argument("--sparsify-rel", type=float, default=0.0)
    ap.add_argument("--sparsify-top", type=int, default=0)
    ap.add_argument("--k-answers", type=int, default=10)
    ap.add_argument("--prune", type=float, default=1e-3)
    ap.add_argument("--n-questions", type=int, default=0, help="evaluation questions (0 = all but calibration)")
    ap.add_argument("--calib-questions", type=int, default=500, help="held out to estimate real score distribution")
    ap.add_argument("--traffic-questions", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    global _KG, _RELAY, _QS, _P, _PK, _SEED, _CAP
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    kg, (cq, cP, cK), (qs, P, pK) = load(a)
    _KG, _QS, _P, _PK, _SEED = kg, qs, P, pK, a.seed
    cross = np.array([bool(q.get("cross_client")) for q in qs])
    print(f"[p2] {os.path.basename(os.path.normpath(a.data_dir))}: {kg.n_clients} clients, {len(qs)} evaluation "
          f"questions ({cross.mean():.2f} cross-client), {len(cq)} calibration questions")

    _CAP = 0
    exact = np.array(_pool_map(_exact_worker, len(qs), a.workers))
    mean = lambda x: float(x.mean()) if x.size else float("nan")  # noqa: E731
    print(f"[exact] Hits@1 {exact.mean():.4f} (cross {mean(exact[cross]):.4f}, local {mean(exact[~cross]):.4f})")

    ints = lambda s: [int(x) for x in s.split(",") if x]  # noqa: E731
    floats = lambda s: [float(x) for x in s.split(",") if x]  # noqa: E731
    rows, cal_cache, capped_cache = [], {}, {}
    pads = []
    for p in a.pads.split(","):
        rule, dummy = p.split(":")
        for mass in (floats(a.mass) if rule == "fixed" else [None]):
            pads.append((rule, dummy, mass))
    for mode, B, d, L, (rule, dummy, mass), gamma, tau in itertools.product(
            a.modes.split(","), ints(a.B), ints(a.d), ints(a.L), pads, floats(a.gamma), floats(a.tau)):
        ts = time.time()
        pad = PadConfig(L=L, mass_per_slot=mass or 0.25, min_dummy_frac=a.min_dummy_frac, dummy_scores="beta",
                        fixed_mass=(rule == "fixed"))
        if L not in capped_cache:
            _CAP = pad.max_real
            capped_cache[L] = np.array(_pool_map(_exact_worker, len(qs), a.workers))
        capped = capped_cache[L]
        if dummy == "empirical":
            key = (mode, L, tau)
            if key not in cal_cache:
                cal_pad = PadConfig(L=L, mass_per_slot=1.0, min_dummy_frac=a.min_dummy_frac, dummy_scores="beta",
                                    fixed_mass=False)
                cal_relay = SketchRelay(kg, B, d, cal_pad, gamma, tau, a.k_answers, a.prune, mode)
                sample = []
                for i in range(len(cq)):
                    st = {"n_real": [], "truncated": [], "scaled": [], "real_scores": []}
                    cal_relay.run(cq[i]["topic_id"], cP[i], cK[i], np.random.default_rng([a.seed, 99, i]), st)
                    sample += st["real_scores"]
                cal_cache[key] = np.concatenate(sample) if sample else np.zeros(0)
            pad = PadConfig(L=L, mass_per_slot=mass or 0.25, min_dummy_frac=a.min_dummy_frac,
                            dummy_scores="empirical", empirical=cal_cache[key], fixed_mass=(rule == "fixed"))
        _RELAY = SketchRelay(kg, B, d, pad, gamma, tau, a.k_answers, a.prune, mode)
        res = _pool_map(_sketch_worker, len(qs), a.workers)
        hits = np.array([r[0] for r in res])
        subs = sum(r[1] for r in res)
        real_scores = np.concatenate([r[5] for r in res]) if res else np.zeros(0)
        tr = traffic(_RELAY, qs, P, pK, a.traffic_questions, a.seed)
        row = {"mode": mode, "B": B, "d": d, "L": L, "pad_rule": rule, "mass_per_slot": mass,
               "M": pad.M if rule == "fixed" else None, "dummy": dummy, "gamma": gamma, "tau": tau,
               "hits1": float(hits.mean()), "hits1_cross": float(hits[cross].mean()) if cross.any() else None,
               "hits1_local": float(hits[~cross].mean()) if (~cross).any() else None,
               "exact": float(exact.mean()), "exact_capped": float(capped.mean()),
               "drop_pts": 100 * float(exact.mean() - hits.mean()),
               "pass_5pts": bool(exact.mean() - hits.mean() <= 0.05),
               "scaled_rate": sum(r[2] for r in res) / max(1, subs),
               "truncated_rate": sum(r[3] for r in res) / max(1, subs),
               "mean_real_per_submission": float(np.mean([r[4] for r in res if r[1]])) if subs else 0.0,
               "real_score_mean": float(real_scores.mean()) if real_scores.size else None,
               "real_score_q": np.quantile(real_scores, [0.1, 0.25, 0.5, 0.75, 0.9]).round(4).tolist()
               if real_scores.size else None}
        row.update({f"traffic_{k}": v for k, v in tr.items()})
        rows.append(row)
        print(f"[{mode} B={B} d={d} L={L} pad={rule}:{dummy}{'' if mass is None else ' mass=' + str(mass)} "
              f"gamma={gamma:g} tau={tau:g}] Hits@1 {row['hits1']:.4f} "
              f"(exact {row['exact']:.4f}, capped {row['exact_capped']:.4f}, drop {row['drop_pts']:.1f} pts) | "
              f"scaled {row['scaled_rate']:.3f} truncated {row['truncated_rate']:.3f} | "
              f"occupancy {tr.get('occupancy_small', float('nan')):.0f} vs {tr.get('occupancy_large', float('nan')):.0f}, "
              f"KS {tr.get('bucket_value_KS', float('nan')):.3f} (null {tr.get('bucket_value_KS_null', float('nan')):.3f}) "
              f"({time.time() - ts:.0f}s)")

    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()})
    summary = {"args": vars(a), "n_questions": len(qs), "n_clients": kg.n_clients,
               "cross_share": float(cross.mean()), "exact_hits1": float(exact.mean()),
               "exact_hits1_cross": float(exact[cross].mean()) if cross.any() else None,
               "configs": rows, "minutes": (time.time() - t0) / 60}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    ok = [r for r in rows if r["pass_5pts"]]
    print(f"\n{len(ok)}/{len(rows)} configurations within 5 points of exact. Wrote {a.out}/summary.json "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
