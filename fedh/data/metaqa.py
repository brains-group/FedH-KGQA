"""MetaQA loading: KB with inverse relations, and vanilla QA files.

Expected layout (the official release, see README):
    <root>/kb.txt                      subject|relation|object
    <root>/{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt
                                       question with [topic entity] <TAB> ans1|ans2|...
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

INV_SUFFIX = "__inv"
_TOPIC = re.compile(r"\[([^\]]+)\]")


@dataclass
class KB:
    ent2id: dict
    id2ent: list
    rel2id: dict
    id2rel: list
    heads: np.ndarray
    rels: np.ndarray
    tails: np.ndarray
    n_facts: int                      # facts before inverses were added
    inverse_of: np.ndarray = field(default=None)  # relation id -> its inverse id

    @property
    def n_ent(self):
        return len(self.id2ent)

    @property
    def n_rel(self):
        return len(self.id2rel)


def load_kb(path: str, add_inverse: bool = True) -> KB:
    ent2id, rel2id = {}, {}
    id2ent, id2rel = [], []
    seen = set()
    H, R, T = [], [], []

    def eid(x):
        if x not in ent2id:
            ent2id[x] = len(id2ent)
            id2ent.append(x)
        return ent2id[x]

    def rid(x):
        if x not in rel2id:
            rel2id[x] = len(id2rel)
            id2rel.append(x)
        return rel2id[x]

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 3:
                continue
            h, r, t = (p.strip() for p in parts)
            if not h or not r or not t:
                continue
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            H.append(eid(h)); R.append(rid(r)); T.append(eid(t))
    n_facts = len(H)
    base_rels = list(id2rel)
    inverse_of = None
    if add_inverse:
        for r in base_rels:
            rid(r + INV_SUFFIX)
        inv = {rel2id[r]: rel2id[r + INV_SUFFIX] for r in base_rels}
        inverse_of = np.empty(len(id2rel), dtype=np.int64)
        for a, b in inv.items():
            inverse_of[a], inverse_of[b] = b, a
        H2 = H + T
        T2 = T + H
        R2 = R + [inv[r] for r in R]
        H, R, T = H2, R2, T2
    return KB(ent2id, id2ent, rel2id, id2rel,
              np.asarray(H, dtype=np.int64), np.asarray(R, dtype=np.int64),
              np.asarray(T, dtype=np.int64), n_facts, inverse_of)


def mask_question(q: str, token: str = "[ENT]") -> str:
    return _TOPIC.sub(token, q, count=1)


def plain_question(q: str) -> str:
    return _TOPIC.sub(lambda m: m.group(1), q)


def load_qa(path: str, kb: KB, hops: int, split: str):
    """Read one vanilla QA file. Questions whose topic entity is not in the KB are dropped."""
    out, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            q, _, ans = line.partition("\t")
            m = _TOPIC.search(q)
            if not m or m.group(1).strip() not in kb.ent2id:
                dropped += 1
                continue
            topic = m.group(1).strip()
            names = [a.strip() for a in ans.split("|") if a.strip()]
            ids = [kb.ent2id[a] for a in names if a in kb.ent2id]
            out.append({
                "qid": f"{hops}hop-{split}-{i}",
                "question": plain_question(q),
                "question_masked": mask_question(q),
                "topic": topic,
                "topic_id": kb.ent2id[topic],
                "answers": ids,
                "answers_missing": [a for a in names if a not in kb.ent2id],
                "K": hops,
            })
    return out, dropped


def qa_path(root: str, hops: int, split: str) -> str:
    return os.path.join(root, f"{hops}-hop", "vanilla", f"qa_{split}.txt")
