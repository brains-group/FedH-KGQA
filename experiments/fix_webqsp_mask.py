"""Mask the topic-entity mention in WebQSP questions where the exact name was not found.

convert_webqsp.py replaces the topic mention only when the Freebase name occurs verbatim in the
question (case-insensitive). Otherwise it appends "[Name]", so the masked question keeps the
original mention (an alias, a surname, a misspelling, an acronym) and clients see it. This
script finds that mention with a fuzzy span match on the original RoG question and masks it,
then writes a copy of the partition whose train / dev / test questions carry the new mask
(every other file is symlinked, so entity ids, owners and labels are unchanged).

Span match, over word tokens (accents stripped, lower case):
  - string similarity of the span and the name >= --min-ratio (SequenceMatcher), or
  - every token of the span is a name token (exact, or similarity >= 0.85 for tokens of 4+
    letters, or a stop word between two name tokens) and the span has a distinctive name token
    of 3+ letters (not a generic geographic word), e.g. a surname, or
  - the span is the acronym of the name (USA, NYC).
The best-scoring span is masked. When nothing matches, the question keeps the appended [ENT];
the leak check below counts how many of those still contain a distinctive name token.

Leak check (before and after): a question leaks when a distinctive token of the topic name
(4+ letters, not generic) appears in the masked question, exactly or with similarity >= 0.85.

  python experiments/fix_webqsp_mask.py --rog-dir data/raw/RoG-webqsp \\
      --part data/processed/webqsp/2hop/C3_rho0.5_balanced_s0 \\
      --out data/processed/webqsp/2hop/C3_rho0.5_balanced_s0_m2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_webqsp import clean, mark  # noqa: E402

STOP = {"the", "of", "and", "a", "an", "in", "on", "at", "for", "to", "de", "la", "le", "du", "von", "van", "der",
        "s", "&", "inc", "co", "ltd"}
QWORDS = {"what", "who", "whom", "whose", "where", "when", "which", "why", "how", "is", "was", "are", "were", "do",
          "does", "did", "has", "have", "had", "can", "could", "will", "would", "the", "of", "in", "on", "for", "to",
          "from", "by", "with", "at", "and", "or", "a", "an", "name", "type", "kind", "year", "time", "play", "live"}
GENERIC = {"city", "state", "states", "county", "river", "island", "islands", "lake", "mount", "mountain", "new",
           "north", "south", "east", "west", "northern", "southern", "eastern", "western", "united", "university",
           "college", "school", "party", "national", "republic", "kingdom", "empire", "church", "saint", "san",
           "santa", "fort", "port", "great", "little", "old", "film", "movie", "book", "series", "show", "team",
           "club", "football", "world", "war", "ocean", "sea", "bay", "park", "street", "high", "district",
           "province", "region", "country", "music", "band", "company", "group", "family", "house", "first"}
_TOK = re.compile(r"\w+", re.UNICODE)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s: str):
    """(normalised token, start, end) with offsets into s."""
    return [(fold(m.group()), m.start(), m.end()) for m in _TOK.finditer(s)]


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def distinctive(name_toks, min_len=3):
    return [t for t in name_toks if len(t) >= min_len and t not in STOP and t not in GENERIC and t not in QWORDS
            and not t.isdigit()]


def tok_match(s: str, t: str) -> bool:
    return s == t or (len(s) >= 4 and len(t) >= 4 and sim(s, t) >= 0.85)


def find_span(question: str, name: str, min_ratio: float = 0.8):
    """Best (start, end, score, rule) character span of the topic mention in question, or None."""
    qt = tokens(question)
    nt = [t for t, _, _ in tokens(name)]
    if not qt or not nt:
        return None
    joined_name = " ".join(nt)
    content = [t for t in nt if t not in STOP]
    # acronyms of the full name and of its leading words (USA, US, NY, UK), with the words they cover
    acr = {}
    for m in range(2, len(content) + 1):
        acr.setdefault("".join(t[0] for t in content[:m]), set(content[:m]))
    if len(nt) >= 2:
        acr.setdefault("".join(t[0] for t in nt), set(nt))
    dist = set(distinctive(nt))
    best = None
    max_len = min(len(nt) + 2, 8)
    for i in range(len(qt)):
        for j in range(i + 1, min(len(qt), i + max_len) + 1):
            span = [t for t, _, _ in qt[i:j]]
            if span[0] in QWORDS or span[-1] in STOP:
                continue
            r = sim(" ".join(span), joined_name)
            rule, score = None, 0.0
            if r >= min_ratio:
                rule, score = "ratio", r
            else:
                pure = all(any(tok_match(s, t) for t in nt) or s in acr or (s in STOP and 0 < k < len(span) - 1)
                           for k, s in enumerate(span))
                hits = [s for s in span if any(tok_match(s, t) for t in dist)]
                covered = {t for t in content if any(tok_match(s, t) for s in span)}
                for s_ in span:
                    covered |= acr.get(s_, set())
                cover = sum(len(t) for t in covered) / max(1, sum(len(t) for t in content))
                if pure and (hits or (j - i >= 2 and cover >= 0.5)):
                    rule, score = "tokens", 0.5 + 0.5 * cover
                elif j == i + 1 and len(span[0]) >= 2 and span[0] in acr:
                    rule, score = "acronym", 0.9
            if rule and (best is None or (score, j - i) > (best[2], best[4])):
                best = (qt[i][1], qt[j - 1][2], score, rule, j - i)
    return None if best is None else best[:4]


def leaks(masked: str, name: str) -> bool:
    dist = [t for t in distinctive([t for t, _, _ in tokens(name)], 4)]
    qt = [t for t, _, _ in tokens(masked.replace("[ENT]", " "))]
    return any(tok_match(s, t) for t in dist for s in qt)


def rog_questions(rog_dir: str, split: str):
    """(original question, topic) for every line convert_webqsp.py wrote, in the same order."""
    import pyarrow.parquet as pq
    out = []
    for f in sorted(glob.glob(os.path.join(rog_dir, "data", f"{split}-*.parquet"))):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=32, columns=["question", "q_entity", "a_entity"]):
            for row in b.to_pylist():
                if not row["q_entity"]:
                    continue
                answers = [clean(x) for x in (row["a_entity"] or []) if clean(x)]
                if not answers:
                    continue
                out.append((re.sub(r"\s+", " ", row["question"]).strip(), clean(row["q_entity"][0])))
    return out


def mask_of(marked: str) -> str:
    return re.sub(r"\[[^\]]+\]", "[ENT]", marked, count=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rog-dir", default="data/raw/RoG-webqsp")
    ap.add_argument("--part", required=True, help="processed WebQSP partition to copy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-ratio", type=float, default=0.8)
    ap.add_argument("--examples", type=int, default=40, help="examples of each kind written to mask_examples.txt")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(a.part):
        if name.endswith(".jsonl") and name.split(".")[0] in ("train", "dev", "test"):
            continue
        dst = os.path.join(a.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(os.path.join(a.part, name)), dst)
    stats, ex = {}, {"fixed": [], "unresolved": []}
    for split, rog in (("train", "train"), ("dev", "validation"), ("test", "test")):
        rq = rog_questions(a.rog_dir, rog)
        if not rq:
            sys.exit(f"no RoG parquet files for split '{rog}' under {a.rog_dir}/data; pass --rog-dir")
        new_mask, kind = [], []
        for q, topic in rq:
            marked, found = mark(q, topic)
            if found:
                new_mask.append(mask_of(marked))
                kind.append("exact")
                continue
            sp = find_span(q, topic, a.min_ratio)
            if sp:
                s, e, score, rule = sp
                new_mask.append(q[:s] + "[ENT]" + q[e:])
                kind.append(rule)
                if len(ex["fixed"]) < a.examples * 3:
                    ex["fixed"].append(f"[{rule} {score:.2f}] {topic} || {q} -> {new_mask[-1]}")
            else:
                new_mask.append(q + " [ENT]")
                kind.append("none")
                if len(ex["unresolved"]) < a.examples * 3:
                    ex["unresolved"].append(f"{topic} || {q}")
        src = os.path.join(a.part, f"{split}.jsonl")
        n = mism = 0
        counts = {}
        leak_before = leak_after = 0
        lines = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                i = int(q["qid"].rsplit("-", 1)[1])
                old = q["question_masked"]
                q_orig, topic = rq[i]
                expect = mask_of(mark(q_orig, topic)[0])
                if expect != old or clean(topic) != q["topic"]:
                    mism += 1
                leak_before += leaks(old, q["topic"])
                q["question_masked"] = new_mask[i]
                leak_after += leaks(new_mask[i], q["topic"])
                counts[kind[i]] = counts.get(kind[i], 0) + 1
                n += 1
                lines.append(json.dumps(q))
        if mism:
            sys.exit(f"[{split}] {mism} of {n} questions do not match the RoG order; nothing written for this split. "
                     f"Check --rog-dir and --part.")
        with open(os.path.join(a.out, f"{split}.jsonl"), "w", encoding="utf-8") as g:
            g.write("\n".join(lines) + "\n")
        stats[split] = {"n": n, "match": counts, "leak_before": leak_before / n, "leak_after": leak_after / n}
        print(f"[{split}] {n} questions | exact {counts.get('exact', 0)}, fuzzy ratio {counts.get('ratio', 0)}, "
              f"name tokens {counts.get('tokens', 0)}, acronym {counts.get('acronym', 0)}, unresolved "
              f"{counts.get('none', 0)} | name visible to clients: {100 * leak_before / n:.1f}% -> "
              f"{100 * leak_after / n:.1f}%")
    with open(os.path.join(a.out, "mask_fix.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(a.out, "mask_examples.txt"), "w", encoding="utf-8") as f:
        f.write("# fixed (rule score) topic || original -> masked\n" + "\n".join(ex["fixed"]) +
                "\n\n# unresolved: topic || original (kept with [ENT] appended)\n" + "\n".join(ex["unresolved"]) + "\n")
    print(f"[done] {a.out} (stats in mask_fix.json, examples in mask_examples.txt)")


if __name__ == "__main__":
    main()
