"""Convert WebQSP (RoG release on Hugging Face, rmanluo/RoG-webqsp) into the MetaQA layout.

Each RoG question comes with its Freebase subgraph (triples within 2 hops of the topic
entities, entity names as node ids, CVT nodes as m.xxx). The union of all subgraphs is
the shared KG that FedH-KGQA partitions across clients.

Output:
  <out>/kb.txt                             head|relation|tail (deduplicated, filtered)
  <out>/2-hop/vanilla/qa_{train,dev,test}.txt   question with [topic] <TAB> ans1|ans2
  <out>/convert_stats.json

The topic is the first q_entity. Its mention in the question is found case-insensitively
and replaced by "[Name]"; if no mention is found, " [Name]" is appended (counted in the stats).
Names are sanitised: '|' -> '/', '[' ']' -> '(' ')', whitespace collapsed.

  python experiments/convert_webqsp.py --rog-dir data/raw/RoG-webqsp --out data/raw/WebQSP
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from array import array

import numpy as np

_WS = re.compile(r"\s+")


def clean(name: str) -> str:
    name = _WS.sub(" ", str(name)).strip()
    return name.replace("|", "/").replace("[", "(").replace("]", ")")


def mark(question: str, topic: str):
    q = _WS.sub(" ", question).strip()
    i = q.lower().find(topic.lower())
    if i >= 0:
        return q[:i] + f"[{topic}]" + q[i + len(topic):], True
    return q + f" [{topic}]", False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rog-dir", default="data/raw/RoG-webqsp")
    ap.add_argument("--out", default="data/raw/WebQSP")
    ap.add_argument("--drop-rel-prefix", default="freebase.valuenotation.,type.object.type,type.type.",
                    help="comma-separated relation prefixes to drop (Freebase bookkeeping)")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    drop = tuple(p for p in a.drop_rel_prefix.split(",") if p)
    ent, rel = {}, {}
    H, R, T = array("q"), array("q"), array("q")
    eid = lambda x: ent.setdefault(x, len(ent))  # noqa: E731
    rid = lambda x: rel.setdefault(x, len(rel))  # noqa: E731
    os.makedirs(os.path.join(a.out, "2-hop", "vanilla"), exist_ok=True)
    stats = {"dropped_relation_prefixes": list(drop), "splits": {}}
    for split, name in (("train", "train"), ("validation", "dev"), ("test", "test")):
        files = sorted(glob.glob(os.path.join(a.rog_dir, "data", f"{split}-*.parquet")))
        n = n_mark = n_nomention = n_noans = n_dropped_rel = 0
        lines = []
        for f in files:
            pf = pq.ParquetFile(f)
            for b in pf.iter_batches(batch_size=a.batch, columns=["question", "q_entity", "a_entity", "graph"]):
                for row in b.to_pylist():
                    for tr in row["graph"] or []:
                        if len(tr) != 3:
                            continue
                        h, r, t = (clean(x) for x in tr)
                        if not h or not r or not t:
                            continue
                        if r.startswith(drop):
                            n_dropped_rel += 1
                            continue
                        H.append(eid(h)); R.append(rid(r)); T.append(eid(t))
                    n += 1
                    if not row["q_entity"]:
                        continue
                    topic = clean(row["q_entity"][0])
                    answers = [clean(x) for x in (row["a_entity"] or []) if clean(x)]
                    if not answers:
                        n_noans += 1
                        continue
                    q, found = mark(row["question"], topic)
                    n_mark += found
                    n_nomention += not found
                    lines.append(f"{q}\t{'|'.join(answers)}")
        with open(os.path.join(a.out, "2-hop", "vanilla", f"qa_{name}.txt"), "w", encoding="utf-8") as g:
            g.write("\n".join(lines) + ("\n" if lines else ""))
        stats["splits"][name] = {"rows": n, "written": len(lines), "mention_found": n_mark,
                                 "mention_appended": n_nomention, "no_answer": n_noans,
                                 "dropped_bookkeeping_triples": n_dropped_rel}
        print(f"[{name}] {n} rows -> {len(lines)} questions (topic mention found in {n_mark}, appended in "
              f"{n_nomention}; {n_noans} without answers)")
    trip = np.unique(np.stack([np.frombuffer(H, np.int64), np.frombuffer(R, np.int64),
                               np.frombuffer(T, np.int64)], 1), axis=0)
    id2e = [None] * len(ent)
    for k, v in ent.items():
        id2e[v] = k
    id2r = [None] * len(rel)
    for k, v in rel.items():
        id2r[v] = k
    with open(os.path.join(a.out, "kb.txt"), "w", encoding="utf-8") as g:
        for h, r, t in trip.tolist():
            g.write(f"{id2e[h]}|{id2r[r]}|{id2e[t]}\n")
    stats.update({"entities": len(ent), "relations": len(rel), "triples": int(trip.shape[0])})
    with open(os.path.join(a.out, "convert_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[kb] {len(ent)} entities, {len(rel)} relations, {trip.shape[0]} unique triples -> {a.out}/kb.txt")


if __name__ == "__main__":
    main()
