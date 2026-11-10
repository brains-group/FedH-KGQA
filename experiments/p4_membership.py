"""P4 attack: QA-pair membership inference against the released reasoner (G1).

The attacker (the server, or any client) holds the final global model, as every party does after
FedAvg. It is given a QA pair (masked question, topic, answers; the relation path follows from
the public schema) and guesses whether the pair was in some client's training set.

Members: training questions with a full local label (they were trained on in every round).
Non-members: test questions with a full local label (same label rule, same distribution, never
trained on). Both get the same score, so the only difference is membership:
  loss attack       score = -(path NLL of the gold relation paths + hop-count cross-entropy)
                    (Yeom et al., 2018: members have lower loss)
  confidence attack score = max over gold paths of the model's path probability
Reported: AUC (0.5 = chance) and the membership advantage max_t (TPR - FPR), on all pairs and on
the pairs whose masked question text is unique (on templated benchmarks many members and
non-members share their exact input after masking, which no attacker can tell apart).

  python experiments/p4_membership.py --runs results/p3v2/metaqa3/<run> results/ablation_dp/metaqa3_dp_eps8 \\
      --out results/p4/membership
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import p1_reasoner as P1  # noqa: E402
from fedh.fed.execute import FederatedKG  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as one half."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    while i < sv.size:
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def advantage(pos: np.ndarray, neg: np.ndarray) -> float:
    """max over thresholds of TPR - FPR (membership advantage)."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    th = np.unique(np.concatenate([pos, neg]))
    tpr = 1 - np.searchsorted(np.sort(pos), th, side="left") / pos.size
    fpr = 1 - np.searchsorted(np.sort(neg), th, side="left") / neg.size
    return float(np.max(tpr - fpr))


@torch.no_grad()
def scores(model, qs, ids, args, device, pad):
    """Per question: (-loss, best gold-path probability)."""
    model.eval()
    out_l, out_c = [], []
    for i in range(0, len(qs), args.eval_batch_size):
        bq = qs[i: i + args.eval_batch_size]
        batch = [{"ids": x, "paths": [p[: args.k_max] for p in q["gold_paths"]], "k": len(q["gold_paths"][0]) - 1}
                 for q, x in zip(bq, ids[i: i + args.eval_batch_size])]
        x, att, paths, pmask, k = P1.collate(batch, pad, args.k_max, device)
        with P1.autocast(device):
            rl, kl = model(x, att)
        logp = F.log_softmax(rl.float(), -1)
        B, Pn, K = paths.shape
        lp = torch.gather(logp.unsqueeze(1).expand(B, Pn, K, logp.size(-1)), 3,
                          paths.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        lp = (lp * (paths >= 0)).sum(-1).masked_fill(pmask == 0, float("-inf"))
        nll = -torch.logsumexp(lp, 1) + F.cross_entropy(kl.float(), k.clamp(min=0), reduction="none")
        out_l.append((-nll).cpu().numpy())
        out_c.append(lp.max(1).values.exp().cpu().numpy())
    return np.concatenate(out_l), np.concatenate(out_c)


def run_one(run_dir, n_max, device, seed):
    with open(os.path.join(run_dir, "metrics.json")) as f:
        meta = json.load(f)
    defaults = {"encoder": "hf", "model_name": "roberta-base", "rel_head": "id", "k_max": 3, "max_len": 48, "lora_r": 8,
                "lora_alpha": 16, "lora_dropout": 0.1, "bow_dim": 256, "eval_batch_size": 256, "local_full_f1": 0.999,
                "max_train": 0, "dp_epsilon": 0.0}
    args = Namespace(**{**defaults, **meta["args"]})
    states = torch.load(os.path.join(run_dir, "trainable_state.pt"), map_location="cpu")
    if "global" not in states:
        return None, "no global model (isolated run)"
    kg = FederatedKG(args.data_dir)
    train = P1.read_jsonl(os.path.join(args.data_dir, "train.jsonl"), args.max_train)
    test = P1.read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    thr = args.local_full_f1
    if args.setting == "centralized":
        ok = lambda q: bool(q.get("gold_paths"))  # noqa: E731
    else:
        ok = lambda q: bool(q.get("gold_paths")) and bool(q.get("local_paths")) and q.get("local_f1", 0) >= thr  # noqa: E731
    mem = [q for q in train if ok(q)]
    non = [q for q in test if ok(q)]
    rng = np.random.default_rng(seed)
    n = min(n_max, len(mem), len(non))
    mem = [mem[i] for i in rng.choice(len(mem), n, replace=False)]
    non = [non[i] for i in rng.choice(len(non), n, replace=False)]
    tokz = P1.Encoderish(args, [q["question_masked"] for q in train])
    model = P1.build_model(args, tokz, kg.n_rel, kg.rel_names, device).to(device)
    model.load_trainable_state({k: v.to(device) for k, v in states["global"].items()})
    lm, cm = scores(model, mem, tokz.encode_all(mem), args, device, tokz.pad_id)
    ln, cn = scores(model, non, tokz.encode_all(non), args, device, tokz.pad_id)
    # questions whose masked text occurs once in the whole training set and never in the test set
    from collections import Counter
    tr_cnt = Counter(q["question_masked"] for q in train)
    te_cnt = Counter(q["question_masked"] for q in test)
    um = np.array([tr_cnt[q["question_masked"]] == 1 and te_cnt[q["question_masked"]] == 0 for q in mem])
    un = np.array([tr_cnt[q["question_masked"]] == 0 and te_cnt[q["question_masked"]] == 1 for q in non])
    shared = float(np.mean([tr_cnt[q["question_masked"]] > 0 for q in non]))
    res = {"run": run_dir, "data": meta["data"], "hops": meta["hops"], "setting": args.setting,
           "dp_epsilon": getattr(args, "dp_epsilon", 0.0), "test_hits1": meta["test"]["hits1"], "n_each": n,
           "nonmember_text_seen_in_train": shared,
           "loss_auc": auc(lm, ln), "loss_adv": advantage(lm, ln),
           "conf_auc": auc(cm, cn), "conf_adv": advantage(cm, cn),
           "unique_n": [int(um.sum()), int(un.sum())],
           "unique_loss_auc": auc(lm[um], ln[un]) if um.sum() >= 30 and un.sum() >= 30 else None,
           "unique_loss_adv": advantage(lm[um], ln[un]) if um.sum() >= 30 and un.sum() >= 30 else None,
           "member_loss_mean": float(-lm.mean()), "nonmember_loss_mean": float(-ln.mean())}
    return res, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (metrics.json + trainable_state.pt)")
    ap.add_argument("--n", type=int, default=5000, help="members and non-members per run (at most)")
    ap.add_argument("--out", default="results/p4/membership")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for r in a.runs:
        ts = time.time()
        if not os.path.exists(os.path.join(r, "trainable_state.pt")):
            print(f"[skip] {r}: no trainable_state.pt")
            continue
        res, why = run_one(r, a.n, device, a.seed)
        if res is None:
            print(f"[skip] {r}: {why}")
            continue
        rows.append(res)
        u = res["unique_loss_auc"]
        print(f"[mi] {res['data']} {res['setting']} eps={res['dp_epsilon'] or 'none'} (test Hits@1 "
              f"{res['test_hits1']:.4f}): loss AUC {res['loss_auc']:.3f} adv {res['loss_adv']:.3f} | conf AUC "
              f"{res['conf_auc']:.3f} | unique-text AUC {'n/a' if u is None else f'{u:.3f}'} "
              f"(n {res['unique_n']}) | non-member text seen in train {100 * res['nonmember_text_seen_in_train']:.0f}% "
              f"| loss member {res['member_loss_mean']:.3f} non-member {res['nonmember_loss_mean']:.3f} "
              f"({time.time() - ts:.0f}s)")
    with open(os.path.join(a.out, "membership.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[done] {a.out}/membership.json")


if __name__ == "__main__":
    main()
