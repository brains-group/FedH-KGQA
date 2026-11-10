"""Convert PathQuestion (IRN release) into the MetaQA layout the P1 pipeline reads.

Input (data/raw/IRN/PathQuestion):
  PQ-2H.txt / PQ-3H.txt / PQL-2H.txt / PQL-3H.txt   question <TAB> ans(alt1/alt2/) <TAB> e0#r1#e1#...#eK[#<end>#ans]
  2H-kb.txt / 3H-kb.txt / PQL2-KB.txt / PQL3-KB.txt  head <TAB> relation <TAB> tail

Output, one MetaQA-style root per variant (each variant has its own KB):
  <out>/<variant>/kb.txt                                  head|relation|tail
  <out>/<variant>/<K>-hop/vanilla/qa_{train,dev,test}.txt  question with [topic] <TAB> ans1|ans2

PathQuestion has no official split; we use the usual random 8:1:1 split with a fixed seed.

  python experiments/convert_pathquestion.py --irn-dir data/raw/IRN/PathQuestion --out data/raw/PathQuestion
"""
from __future__ import annotations

import argparse
import os
import random
import re

VARIANTS = {  # name: (question file, kb file, hops)
    "PQ-2H": ("PQ-2H.txt", "2H-kb.txt", 2),
    "PQ-3H": ("PQ-3H.txt", "3H-kb.txt", 3),
    "PQL-2H": ("PQL-2H.txt", "PQL2-KB.txt", 2),
    "PQL-3H": ("PQL-3H.txt", "PQL3-KB.txt", 3),
}
_ANS = re.compile(r"^([^()]*)\((.*)\)$")


def parse_answers(field: str):
    """'X(X/Y/)' -> [X, Y]. Entity names may themselves contain parentheses, e.g.
    'Venus_(New_version)(Venus_(New_version)/)', so the split point is the '(' after which
    the alternatives list starts with the first answer again."""
    field = field.strip()
    if field.endswith(")"):
        for i, ch in enumerate(field):
            if ch != "(" or i == 0:
                continue
            first, rest = field[:i], field[i + 1:-1]
            if rest == first or rest.startswith(first + "/"):
                alts = [a.strip() for a in rest.split("/") if a.strip()]
                return alts or [first]
        m = _ANS.match(field)
        if m:
            alts = [a.strip() for a in m.group(2).split("/") if a.strip()]
            return alts or [m.group(1).strip()]
    return [field] if field else []


def mark_topic(question: str, topic: str):
    """Put brackets around the topic mention; None if the mention is not found."""
    q = " ".join(question.split())
    toks = q.split(" ")
    for i, t in enumerate(toks):
        if t == topic:
            toks[i] = f"[{topic}]"
            return " ".join(toks)
    idx = q.lower().find(topic.lower())
    if idx < 0:
        return None
    return q[:idx] + "[" + q[idx: idx + len(topic)] + "]" + q[idx + len(topic):]


def convert(irn_dir, out_root, name, qfile, kbfile, hops, seed, ratios):
    out = os.path.join(out_root, name)
    os.makedirs(os.path.join(out, f"{hops}-hop", "vanilla"), exist_ok=True)
    n_kb = 0
    with open(os.path.join(irn_dir, kbfile), encoding="utf-8") as f, \
            open(os.path.join(out, "kb.txt"), "w", encoding="utf-8") as g:
        for line in f:
            parts = line.strip().split("\t") if "\t" in line else line.split()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) != 3:
                continue
            g.write("|".join(parts) + "\n")
            n_kb += 1
    rows, skipped = [], 0
    with open(os.path.join(irn_dir, qfile), encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.rstrip("\n").split("\t") if p.strip()]
            if len(parts) < 3:
                skipped += 1
                continue
            question, ans_field, path = parts[0], parts[1], parts[-1]
            topic = path.strip().split("#")[0]
            answers = parse_answers(ans_field)
            q = mark_topic(question, topic)
            if q is None or not answers:
                skipped += 1
                continue
            rows.append(f"{q}\t{'|'.join(answers)}")
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_tr, n_dv = int(ratios[0] * n), int(ratios[1] * n)
    splits = {"train": rows[:n_tr], "dev": rows[n_tr:n_tr + n_dv], "test": rows[n_tr + n_dv:]}
    for split, rs in splits.items():
        with open(os.path.join(out, f"{hops}-hop", "vanilla", f"qa_{split}.txt"), "w", encoding="utf-8") as g:
            g.write("\n".join(rs) + ("\n" if rs else ""))
    print(f"[{name}] {n_kb} KB triples, {n} questions ({skipped} skipped) -> "
          f"train {len(splits['train'])}, dev {len(splits['dev'])}, test {len(splits['test'])} in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--irn-dir", default="data/raw/IRN/PathQuestion")
    ap.add_argument("--out", default="data/raw/PathQuestion")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for name in a.variants.split(","):
        qfile, kbfile, hops = VARIANTS[name]
        convert(a.irn_dir, a.out, name, qfile, kbfile, hops, a.seed, (0.8, 0.1))


if __name__ == "__main__":
    main()
