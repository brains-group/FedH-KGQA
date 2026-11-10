"""P4: communication and runtime of FedH-KGQA, from finished runs (no GPU).

Training (per dataset, from the federated run): trainable parameters exchanged (LoRA adapters and
heads, fp32), MB per client per round (upload + download), rounds, total MB per client, wall time
(federated and centralized runs, same machine), and relay queries spent on label discovery per
training question. Inference (from p4_relay_attacks.py): MB per query over all clients and the
relay time per query, for our padded sketch relay and for the unpadded raw-ID relay.

  python experiments/p4_cost.py \\
      --row "MetaQA 3-hop" results/p3v2/metaqa3/<fed run> results/p3v2/metaqa3/<central run> results/p4/relay/metaqa3 \\
      --out results/p4/cost.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def load(run):
    with open(os.path.join(run, "metrics.json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", nargs=4, action="append", required=True,
                    metavar=("NAME", "FED_RUN", "CENTRAL_RUN", "RELAY_DIR"))
    ap.add_argument("--out", default="results/p4/cost.json")
    a = ap.parse_args()
    rows = []
    for name, fed, cen, relay in a.row:
        m = load(fed)
        st = torch.load(os.path.join(fed, "trainable_state.pt"), map_location="cpu")["global"]
        n_par = sum(v.numel() for v in st.values())
        mb_round = 2 * n_par * 4 / 1e6
        rounds = m["args"]["rounds"]
        ls = m["label_stats"]
        n_tr = sum(s.get("n", 0) for s in ls.values())
        disc = sum(s.get("discover_queries", 0) + s.get("relay_queries", 0) for s in ls.values())
        row = {"dataset": name, "trainable_params": n_par, "mb_per_client_round": mb_round, "rounds": rounds,
               "local_epochs": m["args"]["local_epochs"], "mb_per_client_total": mb_round * rounds,
               "fed_minutes": m["minutes"], "discovery_queries_per_train_q": disc / max(1, n_tr),
               "test_hits1": m["test"]["hits1"]}
        if cen != "-" and os.path.exists(os.path.join(cen, "metrics.json")):
            row["central_minutes"] = load(cen)["minutes"]
        ap_ = os.path.join(relay, "attacks.json")
        if relay != "-" and os.path.exists(ap_):
            with open(ap_) as f:
                c = json.load(f)["cost"]
            row.update({"query_mb_ours": c["ours"]["bytes_mean"] / 1e6, "query_ms_ours": 1e3 * c["ours"]["secs_mean"],
                        "query_mb_raw": c["naive"]["bytes_mean"] / 1e6, "query_ms_raw": 1e3 * c["naive"]["secs_mean"],
                        "query_mb_raw_p95": c["naive"]["bytes_p95"] / 1e6})
        rows.append(row)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)
    hdr = ("Dataset | Params (M) | MB/client/round | Rounds x epochs | MB/client total | Fed min | Central min | "
           "Disc. queries/q | Query MB ours | Query MB raw IDs | Query ms ours | Query ms raw")
    print(hdr)
    for r in rows:
        g = lambda k, f="{:.2f}": f.format(r[k]) if k in r else "-"  # noqa: E731
        print(f"{r['dataset']} | {r['trainable_params'] / 1e6:.2f} | {r['mb_per_client_round']:.1f} | "
              f"{r['rounds']} x {r['local_epochs']} | {r['mb_per_client_total']:.0f} | {r['fed_minutes']:.0f} | "
              f"{g('central_minutes', '{:.0f}')} | {r['discovery_queries_per_train_q']:.1f} | {g('query_mb_ours')} | "
              f"{g('query_mb_raw', '{:.4f}')} | {g('query_ms_ours', '{:.0f}')} | {g('query_ms_raw', '{:.0f}')}")
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
