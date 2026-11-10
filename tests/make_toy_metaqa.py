"""Generate a small MetaQA-format dataset for smoke tests (no download needed).

Writes <out>/kb.txt and <out>/{1,2}-hop/vanilla/qa_{train,dev,test}.txt with the
same formats as the official release.

  python tests/make_toy_metaqa.py --out data/raw/ToyMetaQA
"""
import argparse
import os
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ToyMetaQA")
    ap.add_argument("--movies", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    directors = [f"director {i}" for i in range(120)]
    actors = [f"actor {i}" for i in range(400)]
    genres = ["Drama", "Comedy", "Horror", "Action", "Romance", "Thriller"]
    years = [str(y) for y in range(1980, 2016)]
    kb = []
    rel = defaultdict(lambda: defaultdict(set))
    for m in range(a.movies):
        mv = f"movie {m}"
        for r, pool, k in (("directed_by", directors, 1), ("starred_actors", actors, 3),
                           ("has_genre", genres, 1), ("release_year", years, 1)):
            for x in rnd.sample(pool, k):
                kb.append((mv, r, x))
                rel[r][mv].add(x)
                rel[r + "_inv"][x].add(mv)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "kb.txt"), "w") as f:
        for h, r, t in kb:
            f.write(f"{h}|{r}|{t}\n")

    def one_hop():
        mv = f"movie {rnd.randrange(a.movies)}"
        r, tmpl = rnd.choice([("directed_by", "who directed [{}]"), ("starred_actors", "who acted in [{}]"),
                              ("has_genre", "what genre is [{}]"), ("release_year", "when was [{}] released")])
        return tmpl.format(mv), rel[r][mv]

    def two_hop():
        kind = rnd.choice(["dir_movies", "actor_codirector", "movie_same_director"])
        if kind == "dir_movies":
            act = rnd.choice(actors)
            ans = {d for m in rel["starred_actors_inv"][act] for d in rel["directed_by"][m]}
            return f"who directed the movies starred by [{act}]", ans
        if kind == "actor_codirector":
            mv = f"movie {rnd.randrange(a.movies)}"
            ans = {x for d in rel["directed_by"][mv] for x in rel["directed_by_inv"][d]} - {mv}
            return f"what other movies did the director of [{mv}] make", ans
        act = rnd.choice(actors)
        ans = {g for m in rel["starred_actors_inv"][act] for g in rel["has_genre"][m]}
        return f"what genres are the films starred by [{act}]", ans

    for hops, gen in ((1, one_hop), (2, two_hop)):
        d = os.path.join(a.out, f"{hops}-hop", "vanilla")
        os.makedirs(d, exist_ok=True)
        for split, n in (("train", 1500), ("dev", 200), ("test", 300)):
            with open(os.path.join(d, f"qa_{split}.txt"), "w") as f:
                k = 0
                while k < n:
                    q, ans = gen()
                    if ans:
                        f.write(q + "\t" + "|".join(sorted(ans)) + "\n")
                        k += 1
    print(f"wrote toy MetaQA to {a.out}: {len(kb)} facts")


if __name__ == "__main__":
    main()
