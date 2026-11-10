"""P0: relay simulation (design revision 4). No training.

Answers three questions before any model code is written:
  1. Collisions: how well does a client read back its frontier from the summed
     count-min sketch at the padded load N*L and fixed mass M, as B and d vary?
     How much of the error comes from dummy entries (padded vs unpadded)?
  2. Traffic shape: with padding, do rounds with small and large real frontiers
     look the same to the server (occupancy, mass, bucket-value histogram)?
  3. Inference DP (G4): what noise keeps the frontier usable, and what epsilon
     does that buy per round and per query?

Modes
  synthetic  random frontiers with a chosen size, score distribution and
             cross-client share. Needs no data.
  metaqa     real frontiers from a P1 partition, following each question's gold
             relation path (an oracle reasoner). Reports answer-set F1 through
             the relay against exact relay, so the gap is the relay's cost alone.

Examples
  python experiments/p0_relay_sim.py synthetic --out results/p0/synthetic
  python experiments/p0_relay_sim.py metaqa \
      --data-dir data/processed/metaqa/2hop/C3_rho0.5_balanced_s0 --out results/p0/metaqa_2hop
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedh.relay.dp import client_noise_std, gaussian_rdp_epsilon, relay_l2_sensitivity  # noqa: E402
from fedh.relay.sketch import (PadConfig, build_sketch, entity_tokens, ks_statistic,  # noqa: E402
                               pad_entries, randomized_round, read_back)

KEY = 0xC0FFEE  # simulated OPRF key held by the querier


def ints(s):
    return [int(float(x)) for x in s.split(",")]


def floats(s):
    return [float(x) for x in s.split(",")]


def aggregate(client_entries, cfg: PadConfig | None, nonce, d, B, n_clients, sigma, gamma, rng):
    """Pad (optionally), sketch, quantise, add distributed noise, sum. Returns (sketch, pad infos)."""
    total = np.zeros((d, B))
    infos = []
    std = client_noise_std(sigma, n_clients, curious_client=True)
    for tok, sc in client_entries:
        if cfg is not None:
            tok, sc, info = pad_entries(tok, sc, cfg, rng)
            infos.append(info)
        sk = build_sketch(tok, sc, nonce, d, B)
        if gamma > 0:
            sk = randomized_round(sk, gamma, rng)
        if std > 0:
            sk = sk + rng.normal(0.0, std, size=sk.shape)
        total += sk
    return total, infos


# ---------------------------------------------------------------- synthetic mode
def synthetic_round(rng, n_clients, owned, frontier_mean, cross, beta, max_real):
    """Real candidates per client: (entity ids, scores). Entity ids are owner*owned + local id."""
    entries = []
    for c in range(n_clients):
        n = int(min(max_real, rng.geometric(1.0 / max(frontier_mean, 1.0))))
        others = [o for o in range(n_clients) if o != c]
        own = np.where((rng.random(n) < cross) & bool(others), rng.choice(others, size=n) if others else c, c)
        ents = own * owned + rng.integers(0, owned, size=n)
        ents = np.unique(ents)
        sc = np.clip(rng.beta(beta[0], beta[1], size=ents.size), 1e-3, 1.0)
        entries.append((ents, sc))
    return entries


def true_scores(entries):
    agg = {}
    for ents, sc in entries:
        for e, s in zip(ents.tolist(), sc.tolist()):
            agg[e] = agg.get(e, 0.0) + s
    return {e: min(1.0, s) for e, s in agg.items()}


def readback_metrics(sketch, nonce, n_clients, owned, owned_tokens, truth, tau):
    """Per-client read-back of every owned entity, compared with the true summed scores."""
    rec_top, mae, fp = [], [], []
    for c in range(n_clients):
        est = np.clip(read_back(sketch, owned_tokens[c], nonce), 0.0, 1.0)
        tr = np.zeros(owned)
        for e, s in truth.items():
            if e // owned == c:
                tr[e % owned] = s
        pos = np.where(tr > 0)[0]
        if pos.size == 0:
            fp.append(int(np.sum(est >= tau)))
            continue
        top = np.argsort(-est)[: pos.size]
        rec_top.append(np.intersect1d(top, pos).size / pos.size)
        mae.append(float(np.mean(np.abs(est[pos] - tr[pos]))))
        neg = np.ones(owned, bool)
        neg[pos] = False
        fp.append(int(np.sum(est[neg] >= tau)))
    return (float(np.mean(rec_top)) if rec_top else float("nan"),
            float(np.mean(mae)) if mae else float("nan"), float(np.mean(fp)))


def run_synthetic(a):
    os.makedirs(a.out, exist_ok=True)
    owned_tokens = [entity_tokens(np.arange(c * a.owned, (c + 1) * a.owned), KEY) for c in range(a.clients)]
    grid = list(itertools.product(ints(a.B), ints(a.d), ints(a.L), floats(a.mass)))
    rows, traffic = [], []
    t0 = time.time()
    for B, d, L, mass in grid:
        cfg = PadConfig(L=L, mass_per_slot=mass, min_dummy_frac=a.min_dummy_frac, dummy_scores=a.dummy_scores)
        delta_l2 = relay_l2_sensitivity(d, fixed_mass=True)
        for sigma in floats(a.sigma):
            acc = {"pad": [], "nopad": []}
            scaled = []
            for t in range(a.trials):
                rng = np.random.default_rng(a.seed * 100003 + t)
                nonce = int(rng.integers(0, 2**62))
                entries = synthetic_round(rng, a.clients, a.owned, a.frontier_mean, a.cross,
                                          (a.beta_a, a.beta_b), cfg.max_real)
                truth = true_scores(entries)
                tok_entries = [(entity_tokens(e, KEY), s) for e, s in entries]
                sk, infos = aggregate(tok_entries, cfg, nonce, d, B, a.clients, sigma, a.gamma, rng)
                acc["pad"].append(readback_metrics(sk, nonce, a.clients, a.owned, owned_tokens, truth, a.tau))
                scaled += [i.scaled for i in infos]
                sk0, _ = aggregate(tok_entries, None, nonce, d, B, a.clients, sigma, a.gamma, rng)
                acc["nopad"].append(readback_metrics(sk0, nonce, a.clients, a.owned, owned_tokens, truth, a.tau))
            nm = sigma / delta_l2 if sigma > 0 else 0.0
            row = {"B": B, "d": d, "L": L, "mass_per_slot": mass, "M": cfg.M, "sigma": sigma,
                   "eps_round": gaussian_rdp_epsilon(nm, 1, a.delta) if sigma > 0 else float("inf"),
                   "eps_query": gaussian_rdp_epsilon(nm, a.rounds, a.delta) if sigma > 0 else float("inf"),
                   "scaled_rate": float(np.mean(scaled))}
            for k in ("pad", "nopad"):
                m = np.array(acc[k], dtype=float)
                row[f"{k}_topn_recall"] = float(np.nanmean(m[:, 0]))
                row[f"{k}_mae"] = float(np.nanmean(m[:, 1]))
                row[f"{k}_false_pos"] = float(np.nanmean(m[:, 2]))
            rows.append(row)
        # traffic shape at sigma = 0: small vs large real frontiers
        vals, occ, mass_tot = {}, {}, {}
        for regime, fmean, off in (("small", 2.0, 0), ("large", 0.8 * cfg.max_real, 0),
                                   ("large_null", 0.8 * cfg.max_real, 10**6)):
            v, o, mt = [], [], []
            for t in range(a.trials):
                rng = np.random.default_rng(a.seed * 7919 + t + off)
                nonce = int(rng.integers(0, 2**62))
                entries = synthetic_round(rng, a.clients, a.owned, fmean, a.cross,
                                          (a.beta_a, a.beta_b), cfg.max_real)
                tok_entries = [(entity_tokens(e, KEY), s) for e, s in entries]
                sk, _ = aggregate(tok_entries, cfg, nonce, d, B, a.clients, 0.0, a.gamma, rng)
                nz = sk[0][sk[0] > 1e-12]
                v.append(nz)
                o.append(nz.size)
                mt.append(float(sk[0].sum()))
            vals[regime], occ[regime], mass_tot[regime] = np.concatenate(v), np.mean(o), np.mean(mt)
        traffic.append({"B": B, "d": d, "L": L, "mass_per_slot": mass,
                        "occupancy_small": occ["small"], "occupancy_large": occ["large"],
                        "mass_small": mass_tot["small"], "mass_large": mass_tot["large"],
                        "bucket_value_KS": ks_statistic(vals["small"], vals["large"]),
                        "bucket_value_KS_null": ks_statistic(vals["large"], vals["large_null"])})
        print(f"[B={B} d={d} L={L} mass={mass}] done ({time.time() - t0:.0f}s)")
    _write_csv(os.path.join(a.out, "sweep.csv"), rows)
    _write_csv(os.path.join(a.out, "traffic.csv"), traffic)
    summary = summarise(rows, traffic, "topn_recall")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"args": vars(a), **summary}, f, indent=2, default=str)
    _print_summary(summary, "top-n recall")


# ---------------------------------------------------------------- MetaQA mode
def load_partition(data_dir, split, n_questions, seed):
    from fedh.data.graph import RelGraph
    owner = []
    with open(os.path.join(data_dir, "entities.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            owner.append(int(line.rstrip("\n").split("\t")[2]))
    owner = np.asarray(owner, dtype=np.int64)
    trip = []
    n_clients = int(owner.max()) + 1
    for c in range(n_clients):
        arr = np.loadtxt(os.path.join(data_dir, "clients", f"client_{c}.tsv"), dtype=np.int64,
                         skiprows=1, ndmin=2)
        trip.append(arr)
    trip = np.concatenate(trip)
    g = RelGraph(owner.size, trip[:, 0], trip[:, 1], trip[:, 2])
    qs = []
    with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
        for line in f:
            q = json.loads(line)
            if q["gold_paths"]:
                qs.append(q)
    rng = np.random.default_rng(seed)
    if n_questions and len(qs) > n_questions:
        qs = [qs[i] for i in rng.permutation(len(qs))[:n_questions]]
    return g, owner, n_clients, qs


def expand_clients(g, owner, n_clients, frontier, r):
    """frontier: per-client (owned ids, scores). Returns per-client clipped candidates."""
    out = []
    for c in range(n_clients):
        ents, sc = frontier[c]
        if ents.size == 0:
            out.append((ents, sc))
            continue
        t, s = g.expand_scored(ents, sc, int(r))
        out.append((t, np.minimum(1.0, s)))
    return out


def cap(ents, sc, k):
    if ents.size <= k:
        return ents, sc
    keep = np.argsort(-sc)[:k]
    return ents[keep], sc[keep]


def exact_relay(g, owner, n_clients, q, max_real=None):
    """Relay with raw IDs (no sketch); optionally with the same per-client cap as the padded relay."""
    e0 = q["topic_id"]
    frontier = [(np.array([e0]), np.array([1.0])) if owner[e0] == c else (np.zeros(0, np.int64), np.zeros(0))
                for c in range(n_clients)]
    for r in q["gold_paths"][0]:
        cand = expand_clients(g, owner, n_clients, frontier, r)
        if max_real:
            cand = [cap(e, s, max_real) for e, s in cand]
        agg = {}
        for e, s in cand:
            for x, y in zip(e.tolist(), s.tolist()):
                agg[x] = agg.get(x, 0.0) + y
        frontier = []
        for c in range(n_clients):
            own = [(x, min(1.0, y)) for x, y in agg.items() if owner[x] == c]
            frontier.append((np.array([x for x, _ in own], np.int64), np.array([y for _, y in own])))
    return frontier


def sketch_relay(g, owner, n_clients, q, owned_ids, owned_tok, cfg, B, d, sigma, gamma, tau, rng):
    e0 = q["topic_id"]
    nonce = int(rng.integers(0, 2**62))
    seed_cfg = PadConfig(L=cfg.L * n_clients, mass_per_slot=cfg.mass_per_slot,
                         min_dummy_frac=cfg.min_dummy_frac, dummy_scores=cfg.dummy_scores)
    sk, _ = aggregate([(entity_tokens([e0], KEY), np.array([1.0]))], seed_cfg, nonce, d, B, 1, 0.0, gamma, rng)
    frontier = _read_frontier(sk, nonce, owned_ids, owned_tok, cfg.max_real, tau, n_clients)
    n_rounds = len(q["gold_paths"][0])
    for r in q["gold_paths"][0]:
        cand = expand_clients(g, owner, n_clients, frontier, r)
        cand = [(entity_tokens(e, KEY), s) for e, s in cand]
        sk, _ = aggregate(cand, cfg, nonce, d, B, n_clients, sigma, gamma, rng)
        frontier = _read_frontier(sk, nonce, owned_ids, owned_tok, cfg.max_real, tau, n_clients)
    return frontier, n_rounds


def _read_frontier(sk, nonce, owned_ids, owned_tok, keep, tau, n_clients):
    out = []
    for c in range(n_clients):
        est = np.clip(read_back(sk, owned_tok[c], nonce), 0.0, 1.0)
        idx = np.where(est > tau)[0]
        idx = idx[np.argsort(-est[idx])][:keep]
        out.append((owned_ids[c][idx], est[idx]))
    return out


def set_f1(frontier, answers, topic=None):
    """Answer-set F1 of the final frontier; the querier drops the topic entity, as MetaQA scoring does."""
    pred = set(int(x) for e, _ in frontier for x in e.tolist()) - ({topic} if topic is not None else set())
    ans = set(answers)
    if not pred or not ans:
        return 0.0, 0.0, 0.0
    hit = len(pred & ans)
    if hit == 0:
        return 0.0, 0.0, 0.0
    p, r = hit / len(pred), hit / len(ans)
    return 2 * p * r / (p + r), p, r


def run_metaqa(a):
    os.makedirs(a.out, exist_ok=True)
    g, owner, n_clients, qs = load_partition(a.data_dir, a.split, a.n_questions, a.seed)
    owned_ids = [np.where(owner == c)[0] for c in range(n_clients)]
    owned_tok = [entity_tokens(ids, KEY) for ids in owned_ids]
    print(f"[metaqa] {len(qs)} questions, {n_clients} clients, "
          f"{[x.size for x in owned_ids]} owned entities; cross-client share "
          f"{np.mean([q['cross_client'] for q in qs]):.2f}")
    exact = np.array([set_f1(exact_relay(g, owner, n_clients, q), q["answers"], q["topic_id"]) for q in qs])
    print(f"[metaqa] exact relay (raw IDs, no cap): F1 {exact[:, 0].mean():.3f}")
    rows = []
    t0 = time.time()
    for B, d, L, mass in itertools.product(ints(a.B), ints(a.d), ints(a.L), floats(a.mass)):
        cfg = PadConfig(L=L, mass_per_slot=mass, min_dummy_frac=a.min_dummy_frac, dummy_scores=a.dummy_scores)
        capped = np.array([set_f1(exact_relay(g, owner, n_clients, q, cfg.max_real), q["answers"], q["topic_id"]) for q in qs])
        delta_l2 = relay_l2_sensitivity(d, fixed_mass=True)
        for sigma in floats(a.sigma):
            rng = np.random.default_rng(a.seed)
            res, rounds = [], 0
            for q in qs:
                fr, rounds = sketch_relay(g, owner, n_clients, q, owned_ids, owned_tok, cfg, B, d,
                                          sigma, a.gamma, a.tau, rng)
                res.append(set_f1(fr, q["answers"], q["topic_id"]))
            res = np.array(res)
            nm = sigma / delta_l2 if sigma > 0 else 0.0
            rows.append({"B": B, "d": d, "L": L, "mass_per_slot": mass, "sigma": sigma,
                         "exact_f1": exact[:, 0].mean(), "exact_capped_f1": capped[:, 0].mean(),
                         "relay_f1": res[:, 0].mean(), "relay_precision": res[:, 1].mean(),
                         "relay_recall": res[:, 2].mean(),
                         "eps_round": gaussian_rdp_epsilon(nm, 1, a.delta) if sigma > 0 else float("inf"),
                         "eps_query": gaussian_rdp_epsilon(nm, a.max_rounds, a.delta) if sigma > 0 else float("inf")})
            print(f"[B={B} d={d} L={L} mass={mass} sigma={sigma}] relay F1 {res[:, 0].mean():.3f} "
                  f"(capped exact {capped[:, 0].mean():.3f}) {time.time() - t0:.0f}s")
    _write_csv(os.path.join(a.out, "sweep.csv"), rows)
    summary = summarise(rows, [], "relay_f1", metaqa=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"args": vars(a), **summary}, f, indent=2, default=str)
    _print_summary(summary, "answer-set F1")


# ---------------------------------------------------------------- reporting
def summarise(rows, traffic, metric, metaqa=False):
    """Per (B, d, L, mass): noiseless quality, and the largest sigma within 5 points of it."""
    key = lambda r: (r["B"], r["d"], r["L"], r["mass_per_slot"])
    out = []
    for k in sorted({key(r) for r in rows}):
        rs = sorted([r for r in rows if key(r) == k], key=lambda r: r["sigma"])
        col = metric if metaqa else f"pad_{metric}"
        base = next((r[col] for r in rs if r["sigma"] == 0), rs[0][col])
        ok = [r for r in rs if r["sigma"] > 0 and r[col] >= base - 0.05]
        best = max(ok, key=lambda r: r["sigma"]) if ok else None
        item = {"B": k[0], "d": k[1], "L": k[2], "mass_per_slot": k[3], "noiseless": base,
                "max_sigma_within_5pts": best["sigma"] if best else None,
                "eps_round_at_that_sigma": best["eps_round"] if best else None,
                "eps_query_at_that_sigma": best["eps_query"] if best else None}
        if not metaqa:
            item["unpadded_noiseless"] = next((r[f"nopad_{metric}"] for r in rs if r["sigma"] == 0), None)
            item["scaled_rate"] = rs[0]["scaled_rate"]
        else:
            item["exact_f1"] = rs[0]["exact_f1"]
            item["exact_capped_f1"] = rs[0]["exact_capped_f1"]
        out.append(item)
    return {"per_config": out, "traffic": traffic}


def _print_summary(summary, name):
    print(f"\nSummary ({name}; 'within 5 pts' is relative to the same config without noise)")
    for it in summary["per_config"]:
        extra = (f" unpadded {it['unpadded_noiseless']:.3f} scaled {it['scaled_rate']:.2f}"
                 if "unpadded_noiseless" in it else
                 f" exact {it['exact_f1']:.3f} capped {it['exact_capped_f1']:.3f}")
        eps = (f"sigma {it['max_sigma_within_5pts']} -> eps/round {it['eps_round_at_that_sigma']:.1f}, "
               f"eps/query {it['eps_query_at_that_sigma']:.1f}" if it["max_sigma_within_5pts"] else "no sigma > 0 within 5 pts")
        print(f"  B={it['B']:>6} d={it['d']} L={it['L']:>3} mass={it['mass_per_slot']}: "
              f"noiseless {it['noiseless']:.3f}{extra} | {eps}")
    for tr in summary["traffic"]:
        print(f"  traffic B={tr['B']} d={tr['d']} L={tr['L']} mass={tr['mass_per_slot']}: occupancy "
              f"{tr['occupancy_small']:.0f} vs {tr['occupancy_large']:.0f}, mass {tr['mass_small']:.1f} vs "
              f"{tr['mass_large']:.1f}, bucket-value KS {tr['bucket_value_KS']:.3f} "
              f"(null {tr['bucket_value_KS_null']:.3f})")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("synthetic", "metaqa"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=f"results/p0/{name}")
        p.add_argument("--B", default="16384,65536,262144", help="sketch widths")
        p.add_argument("--d", default="1,2,3,4", help="hash rows")
        p.add_argument("--L", default="64,128,256", help="padded entries per client per round")
        p.add_argument("--mass", default="0.25,0.5,1.0", help="fixed mass per slot (M = mass * L)")
        p.add_argument("--sigma", default="0,0.05,0.1,0.2,0.4,0.8", help="total std of inference noise")
        p.add_argument("--min-dummy-frac", type=float, default=0.25)
        p.add_argument("--dummy-scores", default="beta", choices=["beta", "even"])
        p.add_argument("--gamma", type=float, default=0.0, help="quantisation scale (0 = off)")
        p.add_argument("--tau", type=float, default=0.05, help="read-back threshold")
        p.add_argument("--delta", type=float, default=1e-5)
        p.add_argument("--seed", type=int, default=0)
    s = sub.choices["synthetic"]
    s.add_argument("--clients", type=int, default=3)
    s.add_argument("--owned", type=int, default=14000, help="entities owned per client")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--frontier-mean", type=float, default=30.0)
    s.add_argument("--cross", type=float, default=0.5, help="share of candidates owned by other clients")
    s.add_argument("--beta-a", type=float, default=2.0)
    s.add_argument("--beta-b", type=float, default=5.0)
    s.add_argument("--trials", type=int, default=10)
    m = sub.choices["metaqa"]
    m.set_defaults(B="65536,262144", d="2,3", L="128,256", mass="0.25,1.0", sigma="0,0.05,0.2")
    m.add_argument("--data-dir", required=True, help="a P1 partition directory")
    m.add_argument("--split", default="test")
    m.add_argument("--n-questions", type=int, default=500)
    m.add_argument("--max-rounds", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "synthetic":
        run_synthetic(a)
    else:
        run_metaqa(a)


if __name__ == "__main__":
    main()
